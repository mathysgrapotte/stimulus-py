"""Default analysis module for stimulus package."""

import math
from typing import Any, Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from torch.utils.data import DataLoader

from stimulus.data.handlertorch import TorchDataset
from stimulus.learner.predict import PredictWrapper


class Analysis:
    """General functions for analysis and plotting.

    TODO automatically set up proper figsize depends on the number of subplots, etc
    """

    @staticmethod
    def get_grid_shape(n: int) -> tuple[int, int]:
        """Calculates rows and columns for a rectangle layout (flexible)."""
        rows = int(math.ceil(math.sqrt(n)))  # Round up the square root for rows
        cols = int(math.ceil(n / rows))  # Calculate columns based on rows
        return rows, cols

    @staticmethod
    def heatmap(
        data: np.ndarray,
        row_labels: list[str],
        col_labels: list[str],
        ax: Any | None = None,
        cbar_kw: dict | None = None,
        cbarlabel: str = "",
        **kwargs: Any,
    ) -> tuple[Any, Any]:
        """Create a heatmap from a numpy array and two lists of labels.

        Parameters
        ----------
        data
            A 2D numpy array of shape (M, N).
        row_labels
            A list or array of length M with the labels for the rows.
        col_labels
            A list or array of length N with the labels for the columns.
        ax
            A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
            not provided, use current axes or create a new one.  Optional.
        cbar_kw
            A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
        cbarlabel
            The label for the colorbar.  Optional.
        **kwargs
            All other arguments are forwarded to `imshow`.
        """
        if ax is None:
            ax = plt.gca()

        if cbar_kw is None:
            cbar_kw = {}

        # Plot the heatmap
        im = ax.imshow(data, **kwargs)

        # Create colorbar
        if ax.figure is not None and hasattr(ax.figure, "colorbar"):
            cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
            cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
        else:
            cbar = None

        # Show all ticks and label them with the respective list entries.
        ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
        ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

        # Let the horizontal axes labeling appear on top.
        ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")

        # Turn spines off and create white grid.
        ax.spines[:].set_visible(False)

        ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)

        return im, cbar

    @staticmethod
    def annotate_heatmap(
        im: Any,
        data: np.ndarray | None = None,
        valfmt: Union[str, StrMethodFormatter] = "{x:.2f}",
        textcolors: tuple[str, str] = ("black", "white"),
        threshold: float | None = None,
        **textkw: Any,
    ) -> list[Any]:
        """A function to annotate a heatmap.

        Parameters
        ----------
        im
            The AxesImage to be labeled.
        data
            Data used to annotate.  If None, the image's data is used.  Optional.
        valfmt
            The format of the annotations inside the heatmap.  This should either
            use the string format method, e.g. "$ {x:.2f}", or be a
            `matplotlib.ticker.Formatter`.  Optional.
        textcolors
            A pair of colors.  The first is used for values below a threshold,
            the second for those above.  Optional.
        threshold
            Value in data units according to which the colors from textcolors are
            applied.  If None (the default) uses the middle of the colormap as
            separation.  Optional.
        **kwargs
            All other arguments are forwarded to each call to `text` used to create
            the text labels.
        """
        if not isinstance(data, (list, np.ndarray)):
            data = im.get_array()

        # Normalize the threshold to the images color range.
        threshold = im.norm(threshold) if threshold is not None else im.norm(data.max()) / 2.0

        # Set default alignment to center, but allow it to be
        # overwritten by textkw.
        kw = {"horizontalalignment": "center", "verticalalignment": "center"}
        kw.update(textkw)

        # Get the formatter in case a string is supplied
        if isinstance(valfmt, str):
            valfmt = StrMethodFormatter(valfmt)

        # Loop over the data and create a `Text` for each "pixel".
        # Change the text's color depending on the data.
        texts = []
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
                text = im.axes.text(j, i, valfmt(data[i, j]), **kw)
                texts.append(text)

        return texts


class AnalysisPerformanceTune(Analysis):
    """Report the performance during tuning.

    TODO maybe instead of reporting one pdf for one model with all metrics,
    report one pdf for all models with one metric.
    TODO or maybe one pdf for all models with all metrics, colored by model. One for train, one for val.
    """

    def __init__(self, results_path: str) -> None:
        """Initialize the AnalysisPerformanceTune class."""
        super().__init__()
        self.results = pd.read_csv(results_path)

    def plot_metric_vs_iteration(
        self,
        metrics: list,
        figsize: tuple = (10, 10),
        output: str | None = None,
    ) -> None:
        """Plot metrics vs iteration for training and validation."""
        # create figure
        rows, cols = self.get_grid_shape(len(metrics))
        fig, axs = plt.subplots(rows, cols, figsize=figsize)

        # plot each metric
        for i, ax in enumerate(axs.flat):
            if i >= len(metrics):
                ax.axis("off")
                continue
            self.plot_metric_vs_iteration_per_metric(axs.flat[i], metrics[i])

        # add legend
        # axs.flat[0].legend()
        handles, labels = axs[0, 0].get_legend_handles_labels()  # Get handles and labels from one subplot
        plt.legend(handles, labels, loc="upper left")  # Adjust location as needed

        # save plot
        plt.tight_layout()
        if output:
            plt.savefig(output)
        plt.show()

    def plot_metric_vs_iteration_per_metric(self, ax: Any, metric: str) -> Any:
        """Plot the metric vs the iteration."""
        # plot training performance
        ax.plot(
            self.results.training_iteration,
            self.results["train_" + metric],
            c="blue",
            label="train",
        )

        # plot validation performance
        ax.plot(
            self.results.training_iteration,
            self.results["val_" + metric],
            c="orange",
            label="val",
        )

        # TODO set x-axis labels into integer
        # plt.xticks(range(min(self.results.training_iteration), max(self.results.training_iteration)))

        # add labels
        ax.set_xlabel("epoch")
        ax.set_ylabel(metric)

        return ax


class AnalysisRobustness(Analysis):
    """Report the robustness of the models."""

    def __init__(self, metrics: list, experiment: object, batch_size: int) -> None:
        """Initialize the AnalysisRobustness class."""
        super().__init__()
        self.metrics = metrics
        self.experiment = experiment
        self.batch_size = batch_size

    def get_performance_table(self, names: list, model_list: dict, data_list: list) -> pd.DataFrame:
        """Compute the performance metrics of each model on each dataset.

        Args:
            names: List of names that identifies each model.
            model_list: Dictionary of models in same order as data_list.
            data_list: List of datasets used for training.

        Returns:
            DataFrame containing performance metrics.
        """
        # check same length
        if (len(names) != len(model_list)) and (len(names) != len(data_list)):
            raise ValueError("The length of the names, model_list and data_list should be the same.")

        # initialize
        df = pd.DataFrame()
        model_names = []

        # for each model, get the performance table, and concat
        for i, model in enumerate(model_list):
            df = pd.concat([df, self.get_performance_table_for_one_model(names, model, data_list)])
            model_names += [names[i]] * len(data_list)
        df["model"] = model_names

        return df

    def get_performance_table_for_one_model(self, names: list, model: object, data_list: list) -> pd.DataFrame:
        """Compute the performance table of one model on each dataset."""
        df = pd.DataFrame()
        for data_path in data_list:  # for each data, get the performance metrics, and concat
            # initialize the dataframe keeping the original order, aka no shuffle
            dataloader = DataLoader(
                TorchDataset(data_path, self.experiment, split=2),
                batch_size=self.batch_size,
                shuffle=False,
            )
            metric_values = PredictWrapper(model, dataloader).compute_metrics(self.metrics)
            df = pd.concat([df, pd.DataFrame(metric_values, index=[0])])
        df["data"] = names
        return df

    def get_average_performance_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute the average performance of each model on each dataset.

        Args:
            df: DataFrame containing the performance table.

        Returns:
            DataFrame with averaged metrics.
        """
        df = df[[*self.metrics, "model"]]  # Use list unpacking instead of concatenation
        return df.groupby(["model"]).mean().reset_index()

    def plot_performance_heatmap(self, df: pd.DataFrame, figsize: tuple = (10, 10), output: str | None = None) -> None:
        """Plot the performance of each model on each dataset."""
        # create figure
        rows, cols = self.get_grid_shape(len(self.metrics))
        fig, axs = plt.subplots(rows, cols, figsize=figsize)

        # if there is only one plot plot.sublots will output a simple list, while if there are more than one it will return a list of lists. there is the need to unify the two cases. following line does this
        if not isinstance(axs, np.ndarray):
            axs = np.array([axs])

        for i, ax in enumerate(axs.flat):
            if i >= len(self.metrics):
                ax.axis("off")
                continue

            # reshape the data frame into the matrix for one metric
            mat = df[["model", "data", self.metrics[i]]]
            mat = mat.pivot(index="model", columns="data", values=self.metrics[i])

            # plot heatmap
            im, cbar = self.heatmap(mat, mat.index, mat.columns, ax=ax, cmap="YlGn", cbarlabel=self.metrics[i])
            self.annotate_heatmap(im, valfmt="{x:.2f}")  # Don't assign to unused variable

        # save plot
        plt.tight_layout()
        if output:
            plt.savefig(output)
        plt.show()

    def plot_delta_performance(
        self,
        metric: str,
        df: pd.DataFrame,
        figsize: tuple = (10, 10),
        output: str | None = None,
    ) -> None:
        """Plot the delta performance of each model on each dataset."""
        # create figure
        rows, cols = self.get_grid_shape(len(df["model"].unique()))
        fig, axs = plt.subplots(rows, cols, figsize=figsize)

        # if there is only one plot plot.sublots will output <class 'matplotlib.axes._axes.Axes'>, while if there are more than one it will return a np.ndarray. there is the need to unify the two cases. following line does this
        if not isinstance(axs, np.ndarray):
            axs = np.array([axs])

        # plot each model
        for i, ax in enumerate(axs.flat):
            if i >= len(df["model"].unique()):
                ax.axis("off")
                continue
            self.plot_delta_performance_for_one_model(ax, metric, df, df["model"].unique()[i])

        # set common y limits
        ymin = min([ax.get_ylim()[0] for ax in axs.flat])
        ymax = max([ax.get_ylim()[1] for ax in axs.flat])
        for ax in axs.flat:
            spacer = abs(ymin - ymax)
            spacer = spacer * 0.01
            ax.set_ylim(ymin - spacer, ymax + spacer)

        # save plot
        plt.tight_layout()
        if output:
            plt.savefig(output)
        plt.show()

    def plot_delta_performance_for_one_model(self, ax: Any, metric: str, df: pd.DataFrame, model_name: str) -> Any:
        """Plot the delta performance of one model."""
        df = self.parse_delta_performance_for_one_model(metric, df, model_name)

        # plot a barplot with positive negative values for each row
        # TODO use different colors for positive and negative values
        df = df.set_index("data")
        df.plot(kind="bar", ax=ax, stacked=True)

        ax.set_xlabel("")
        ax.get_legend().remove()
        ax.set_title(model_name)

        return ax

    def parse_delta_performance_for_one_model(self, metric: str, df: pd.DataFrame, model_name: str) -> pd.DataFrame:
        """Compute the delta performance of one model."""
        # filter data frame
        df = df[["data", "model", metric]]
        df = df[df["model"] == model_name]

        # compute the delta performance between each row vs the reference
        reference_row = df.loc[df["data"] == model_name]
        df[metric] = -df[metric].sub(reference_row[metric])

        return df
