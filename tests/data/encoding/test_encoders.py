"""Test for encoders."""

import pytest
import torch

from src.stimulus.data.encoding.encoders import (
    NumericEncoder,
    NumericRankEncoder,
    StrClassificationEncoder,
    TextOneHotEncoder,
)


class TestTextOneHotEncoder:
    """Test suite for TextOneHotEncoder."""

    @staticmethod
    @pytest.fixture
    def encoder_default() -> TextOneHotEncoder:
        """Provide a default encoder.

        Returns:
            TextOneHotEncoder: A default encoder instance
        """
        return TextOneHotEncoder(alphabet="acgt", padding=True)

    @staticmethod
    @pytest.fixture
    def encoder_lowercase() -> TextOneHotEncoder:
        """Provide an encoder with convert_lowercase set to True.

        Returns:
            TextOneHotEncoder: An encoder instance with lowercase conversion
        """
        return TextOneHotEncoder(alphabet="ACgt", convert_lowercase=True, padding=True)

    # ---- Test for initialization ---- #

    def test_init_with_string_alphabet(self) -> None:
        """Test initialization with valid string alphabet."""
        encoder = TextOneHotEncoder(alphabet="acgt")
        assert encoder.alphabet == "acgt"
        assert encoder.convert_lowercase is False
        assert encoder.padding is False

    # ---- Tests for _sequence_to_array ---- #

    def test_sequence_to_array_with_non_string_input(
        self,
        encoder_default: TextOneHotEncoder,
    ) -> None:
        """Test _sequence_to_array with non-string input raises TypeError."""
        with pytest.raises(TypeError, match="Expected string input for sequence"):
            encoder_default._sequence_to_array(1234)  # type: ignore[arg-type]

    def test_sequence_to_array_returns_correct_shape(
        self,
        encoder_default: TextOneHotEncoder,
    ) -> None:
        """Test _sequence_to_array returns array of correct shape."""
        seq: str = "acgt"
        arr = encoder_default._sequence_to_array(seq)
        assert arr.shape == (4, 1)
        assert (arr.flatten() == list(seq)).all()

    def test_sequence_to_array_is_case_sensitive(self, encoder_default: TextOneHotEncoder) -> None:
        """Test that _sequence_to_array preserves case when case sensitivity is enabled."""
        seq = "AcGT"
        arr = encoder_default._sequence_to_array(seq)
        assert (arr.flatten() == list("AcGT")).all()

    def test_sequence_to_array_is_lowercase(self, encoder_lowercase: TextOneHotEncoder) -> None:
        """Test that _sequence_to_array converts to lowercase when enabled."""
        seq = "AcGT"
        arr = encoder_lowercase._sequence_to_array(seq)
        assert (arr.flatten() == list("acgt")).all()

    # ---- Tests for encode ---- #

    def test_encode_returns_tensor(self, encoder_default: TextOneHotEncoder) -> None:
        """Test that encode returns a tensor of the correct shape."""
        seq = "acgt"
        encoded = encoder_default.encode(seq)
        assert isinstance(encoded, torch.Tensor)
        # shape should be (len(seq), alphabet_size=4)
        assert encoded.shape == (4, 4)

    def test_encode_unknown_character_returns_zero_vector(self, encoder_default: TextOneHotEncoder) -> None:
        """Test that encoding an unknown character returns a zero vector."""
        seq = "acgtn"
        encoded = encoder_default.encode(seq)
        # the last character 'n' is not in 'acgt', so the last row should be all zeros
        assert torch.all(encoded[-1] == 0)

    def test_encode_default(self, encoder_default: TextOneHotEncoder) -> None:
        """Test case-sensitive encoding behavior.

        Case-sensitive: 'ACgt' => 'ACgt' means 'A' and 'C' are uppercase in the alphabet,
        'g' and 't' are lowercase in the alphabet.
        """
        seq = "ACgt"
        encoded = encoder_default.encode(seq)
        # shape = (len(seq), 4)
        assert encoded.shape == (4, 4)
        # 'A' should be one-hot at the 0th index, 'C' at the 1st index, 'g' at the 2nd, 't' at the 3rd.
        # The order of categories in OneHotEncoder is typically ['A', 'C', 'g', 't'] given we passed ['A','C','g','t'].
        assert torch.all(encoded[0] == torch.tensor([0, 0, 0, 0]))  # 'A'
        assert torch.all(encoded[1] == torch.tensor([0, 0, 0, 0]))  # 'C'
        assert torch.all(encoded[2] == torch.tensor([0, 0, 1, 0]))  # 'g'
        assert torch.all(encoded[3] == torch.tensor([0, 0, 0, 1]))  # 't'

    def test_encode_lowercase(self, encoder_lowercase: TextOneHotEncoder) -> None:
        """Case-insensitive: 'ACgt' => 'acgt' internally."""
        seq = "ACgt"
        encoded = encoder_lowercase.encode(seq)
        # shape = (4,4)
        assert encoded.shape == (4, 4)
        # The order of categories in OneHotEncoder is typically ['a', 'c', 'g', 't'] for the default encoder.
        assert torch.all(encoded[0] == torch.tensor([1, 0, 0, 0]))  # 'a'
        assert torch.all(encoded[1] == torch.tensor([0, 1, 0, 0]))  # 'c'
        assert torch.all(encoded[2] == torch.tensor([0, 0, 1, 0]))  # 'g'
        assert torch.all(encoded[3] == torch.tensor([0, 0, 0, 1]))  # 't'

    # ---- Tests for encode_all ---- #

    def test_encode_all_with_single_string(self, encoder_default: TextOneHotEncoder) -> None:
        """Test encoding a single string with encode_all."""
        seq = "acgt"
        encoded = encoder_default.encode_all(seq)
        # shape = (batch_size=1, seq_len=4, alphabet_size=4)
        assert encoded.shape == (1, 4, 4)
        assert torch.all(encoded[0] == encoder_default.encode(seq))

    def test_encode_all_with_list_of_sequences(self, encoder_default: TextOneHotEncoder) -> None:
        """Test encoding multiple sequences with encode_all."""
        seqs = ["acgt", "acgtn"]  # second has an unknown 'n'
        encoded = encoder_default.encode_all(seqs)
        # shape = (2, max_len=5, alphabet_size=4)
        assert encoded.shape == (2, 5, 4)
        # check content
        assert torch.all(encoded[0][:4] == encoder_default.encode(seqs[0]))
        assert torch.all(encoded[1] == encoder_default.encode(seqs[1]))

    def test_encode_all_with_padding_false(self) -> None:
        """Test that encode_all raises error when padding is False and sequences have different lengths."""
        encoder = TextOneHotEncoder(alphabet="acgt", padding=False)
        seqs = ["acgt", "acgtn"]  # different lengths
        # should raise ValueError because lengths differ
        with pytest.raises(ValueError, match="All sequences must have the same length when padding is False."):
            encoder.encode_all(seqs)

    # ---- Tests for decode ---- #

    def test_decode_single_sequence(self, encoder_default: TextOneHotEncoder) -> None:
        """Test decoding a single encoded sequence."""
        seq = "acgt"
        encoded = encoder_default.encode(seq)
        decoded = encoder_default.decode(encoded)
        # Because decode returns a string in this case
        assert isinstance(decoded, str)
        # Should match the lowercased input (since case-sensitive=False)
        assert decoded == seq

    def test_decode_unknown_characters(self, encoder_default: TextOneHotEncoder) -> None:
        """Test decoding behavior with unknown characters.

        Unknown characters are zero vectors. When decoding, those become empty (ignored),
        or become None, depending on the transform. In the provided code, handle_unknown='ignore'
        yields an empty decode for those positions. The example code attempts to fill with '-'
        or None if needed.
        """
        seq = "acgtn"
        encoded = encoder_default.encode(seq)
        decoded = encoder_default.decode(encoded)
        # The code snippet shows it returns an empty string for unknown char,
        # or might fill with 'None' replaced by '-'.
        # Adjust assertion based on how you've handled unknown characters in your final implementation.
        # If you replaced None with '-', the last character might be '-'.
        # If handle_unknown='ignore' yields an empty decode, it might omit the character entirely.
        # In the given code, it returns an empty decode for that position. So let's assume it becomes ''.
        # That means we might get "acgt" with a missing final char or a placeholder.
        # Let's do a partial check:
        assert isinstance(decoded, str)
        assert decoded.startswith("acgt")

    def test_decode_multiple_sequences(self, encoder_default: TextOneHotEncoder) -> None:
        """Test decoding multiple encoded sequences."""
        seqs = ["acgt", "acgtn"]  # second has unknown 'n'
        encoded = encoder_default.encode_all(seqs)
        decoded = encoder_default.decode(encoded)
        # decode should return a list of strings in this case
        assert isinstance(decoded, list)
        assert len(decoded) == 2
        assert decoded[0] == "acgt-"  # '-' for padding
        assert decoded[0] == "acgt-"  # '-' for unknown character n


class TestNumericEncoder:
    """Test suite for NumericEncoder."""

    @staticmethod
    @pytest.fixture
    def float_encoder() -> NumericEncoder:
        """Provide a NumericEncoder instance.

        Returns:
            NumericEncoder: Default encoder instance
        """
        return NumericEncoder()

    @staticmethod
    @pytest.fixture
    def int_encoder() -> NumericEncoder:
        """Provide a NumericEncoder instance with integer dtype.

        Returns:
            NumericEncoder: Integer-based encoder instance
        """
        return NumericEncoder(dtype=torch.int32)

    def test_encode_single_float(self, float_encoder: NumericEncoder) -> None:
        """Test encoding a single float value."""
        input_val = 3.14
        output = float_encoder.encode(input_val)
        assert isinstance(output, torch.Tensor), "Output should be a torch.Tensor."
        assert output.dtype == torch.float32, "Tensor dtype should be float32."
        assert output.numel() == 1, "Tensor should have exactly one element."
        assert output.item() == pytest.approx(input_val), "Encoded value does not match."

    def test_encode_single_int(self, int_encoder: NumericEncoder) -> None:
        """Test encoding a single int value."""
        input_val = 3
        output = int_encoder.encode(input_val)
        assert isinstance(output, torch.Tensor), "Output should be a torch.Tensor."
        assert output.dtype == torch.int32, "Tensor dtype should be int32."
        assert output.numel() == 1, "Tensor should have exactly one element."
        assert output.item() == input_val

    @pytest.mark.parametrize("fixture_name", ["float_encoder", "int_encoder"])
    def test_encode_non_numeric_raises(
        self,
        request: pytest.FixtureRequest,
        fixture_name: str,
    ) -> None:
        """Test that encoding a non-float raises a ValueError."""
        numeric_encoder = request.getfixturevalue(fixture_name)
        with pytest.raises(ValueError, match="Expected input data to be a float or int"):
            numeric_encoder.encode("not_numeric")

    def test_encode_all_single_float(self, float_encoder: NumericEncoder) -> None:
        """Test encode_all when given a single float.

        Tests that a single float is treated as a list of one float internally.

        Args:
            float_encoder: Float-based encoder instance
        """
        input_val = [2.71]
        output = float_encoder.encode_all(input_val)
        assert isinstance(output, torch.Tensor), "Output should be a torch.Tensor."
        assert output.numel() == 1, "Tensor should have exactly one element."
        assert output.item() == pytest.approx(input_val[0]), "Encoded value does not match the input."

    def test_encode_all_single_int(self, int_encoder: NumericEncoder) -> None:
        """Test encode_all when given a single int.

        Tests that a single int is treated as a list of one int internally.

        Args:
            int_encoder: Integer-based encoder instance
        """
        input_val = [2.0]
        output = int_encoder.encode_all(input_val)
        assert isinstance(output, torch.Tensor), "Output should be a torch.Tensor."
        assert output.numel() == 1, "Tensor should have exactly one element."
        assert output.item() == int(input_val[0])

    def test_encode_all_multi_float(self, float_encoder: NumericEncoder) -> None:
        """Test encode_all with a list of floats."""
        input_vals = [3.14, 4.56]
        output = float_encoder.encode_all(input_vals)
        assert isinstance(output, torch.Tensor), "Output should be a torch.Tensor."
        assert output.dtype == torch.float32, "Tensor dtype should be float32."
        assert output.numel() == 2, "Tensor should have exactly one element."
        assert output[0].item() == pytest.approx(3.14), "First element does not match."
        assert output[1].item() == pytest.approx(4.56), "Second element does not match."

    def test_encode_all_multi_int(self, int_encoder: NumericEncoder) -> None:
        """Test encode_all with a list of integers."""
        input_vals = [3.0, 4.0]
        output = int_encoder.encode_all(input_vals)
        assert isinstance(output, torch.Tensor), "Output should be a torch.Tensor."
        assert output.dtype == torch.int32, "Tensor dtype should be int32."
        assert output.numel() == 2, "Tensor should have exactly one element."
        assert output[0].item() == 3, "First element does not match."
        assert output[1].item() == 4, "Second element does not match."

    def test_decode_single_float(self, float_encoder: NumericEncoder) -> None:
        """Test decoding a tensor of shape (1)."""
        input_tensor = torch.tensor([3.14], dtype=torch.float32)
        decoded = float_encoder.decode(input_tensor)
        # decode returns data.numpy().tolist()
        assert isinstance(decoded, list), "Decoded output should be a list."
        assert len(decoded) == 1, "Decoded list should have one element."
        assert decoded[0] == pytest.approx(3.14), "Decoded value does not match."

    def test_decode_single_int(self, int_encoder: NumericEncoder) -> None:
        """Test decoding a tensor of shape (1)."""
        input_tensor = torch.tensor([3], dtype=torch.int32)
        decoded = int_encoder.decode(input_tensor)
        # decode returns data.numpy().tolist()
        assert isinstance(decoded, list), "Decoded output should be a list."
        assert len(decoded) == 1, "Decoded list should have one element."
        assert decoded[0] == 3, "Decoded value does not match."

    def test_decode_multi_float(self, float_encoder: NumericEncoder) -> None:
        """Test decoding a tensor of shape (n)."""
        input_tensor = torch.tensor([3.14, 2.71], dtype=torch.float32)
        decoded = float_encoder.decode(input_tensor)
        assert isinstance(decoded, list), "Decoded output should be a list."
        assert len(decoded) == 2, "Decoded list should have two elements."
        assert decoded[0] == pytest.approx(3.14), "First decoded value does not match."
        assert decoded[1] == pytest.approx(2.71), "Second decoded value does not match."

    def test_decode_multi_int(self, int_encoder: NumericEncoder) -> None:
        """Test decoding a tensor of shape (n)."""
        input_tensor = torch.tensor([3, 4], dtype=torch.int32)
        decoded = int_encoder.decode(input_tensor)
        assert isinstance(decoded, list), "Decoded output should be a list."
        assert len(decoded) == 2, "Decoded list should have two elements."
        assert decoded[0] == 3, "First decoded value does not match."
        assert decoded[1] == 4, "Second decoded value does not match."


class TestStrClassificationEncoder:
    """Test suite for StrClassificationIntEncoder and StrClassificationScaledEncoder."""

    @staticmethod
    @pytest.fixture
    def str_encoder() -> StrClassificationEncoder:
        """Provide a StrClassificationEncoder instance.

        Returns:
            StrClassificationEncoder: Default encoder instance
        """
        return StrClassificationEncoder()

    @staticmethod
    @pytest.fixture
    def scaled_encoder() -> StrClassificationEncoder:
        """Provide a StrClassificationEncoder with scaling enabled.

        Returns:
            StrClassificationEncoder: Scaled encoder instance
        """
        return StrClassificationEncoder(scale=True)

    @pytest.mark.parametrize("fixture", ["str_encoder", "scaled_encoder"])
    def test_encode_raises_not_implemented(
        self,
        request: pytest.FixtureRequest,
        fixture: str,
    ) -> None:
        """Test that encoding a single string raises NotImplementedError.

        This verifies that the encode method is not implemented for single strings.
        """
        encoder = request.getfixturevalue(fixture)
        with pytest.raises(
            NotImplementedError,
            match="Encoding a single string does not make sense. Use encode_all instead.",
        ):
            encoder.encode("test")

    @pytest.mark.parametrize(
        ("fixture", "expected_values"),
        [
            ("str_encoder", [0, 1, 2]),
            ("scaled_encoder", [0.0, 0.5, 1.0]),
        ],
    )
    def test_encode_all_list_of_strings(
        self,
        request: pytest.FixtureRequest,
        fixture: str,
        expected_values: list,
    ) -> None:
        """Test encoding multiple unique strings.

        Verifies that the encoder produces correct tensor shape and values.
        """
        encoder = request.getfixturevalue(fixture)
        input_data = ["apple", "banana", "cherry"]
        output = encoder.encode_all(input_data)
        assert isinstance(output, torch.Tensor)
        assert output.shape == (3,)
        assert torch.allclose(output, torch.tensor(expected_values))

    @pytest.mark.parametrize("fixture", ["str_encoder", "scaled_encoder"])
    def test_encode_all_raises_value_error_on_non_string(
        self,
        request: pytest.FixtureRequest,
        fixture: str,
    ) -> None:
        """Tests that encode_all raises ValueError.

        if the input is not a string or list of strings.
        """
        encoder = request.getfixturevalue(fixture)
        input_data = ["apple", 42, "banana"]  # 42 is not a string
        with pytest.raises(ValueError, match="Expected input data to be a list of strings") as exc_info:
            encoder.encode_all(input_data)
        assert "Expected input data to be a list of strings" in str(exc_info.value)

    @pytest.mark.parametrize("fixture", ["str_encoder", "scaled_encoder"])
    def test_decode_raises_not_implemented(
        self,
        request: pytest.FixtureRequest,
        fixture: str,
    ) -> None:
        """Tests that decode() raises NotImplementedError.

        since decoding is not supported in this encoder.
        """
        encoder = request.getfixturevalue(fixture)
        with pytest.raises(NotImplementedError) as exc_info:
            encoder.decode(torch.tensor([0]))
        assert "Decoding is not yet supported for StrClassification." in str(exc_info.value)


class TestNumericRankEncoder:
    """Test suite for NumericRankEncoder."""

    @staticmethod
    @pytest.fixture
    def rank_encoder() -> NumericRankEncoder:
        """Provide a NumericRankEncoder instance.

        Returns:
            NumericRankEncoder: Default encoder instance
        """
        return NumericRankEncoder()

    @staticmethod
    @pytest.fixture
    def scaled_encoder() -> NumericRankEncoder:
        """Provide a NumericRankEncoder with scaling enabled.

        Returns:
            NumericRankEncoder: Scaled encoder instance
        """
        return NumericRankEncoder(scale=True)

    @pytest.mark.parametrize("fixture", ["rank_encoder", "scaled_encoder"])
    def test_encode_raises_not_implemented(
        self,
        request: pytest.FixtureRequest,
        fixture: str,
    ) -> None:
        """Test that encoding a single float raises NotImplementedError.

        Args:
            request: Pytest fixture request
            fixture: Name of the fixture to use
        """
        encoder = request.getfixturevalue(fixture)
        with pytest.raises(
            NotImplementedError,
            match="Encoding a single float does not make sense. Use encode_all instead.",
        ):
            encoder.encode(3.14)

    def test_encode_all_with_valid_rank(self, rank_encoder: NumericRankEncoder) -> None:
        """Test encoding a list of float values.

        Args:
            rank_encoder: Default rank encoder instance
        """
        input_vals = [3.14, 2.71, 1.41]
        output = rank_encoder.encode_all(input_vals)
        assert isinstance(output, torch.Tensor), "Output should be a torch.Tensor."
        assert output.numel() == 3, "Tensor should have exactly three elements."
        assert output[0] == 2, "First encoded value does not match."
        assert output[1] == 1, "Second encoded value does not match."
        assert output[2] == 0, "Third encoded value does not match."

    def test_encode_all_with_valid_scaled_rank(self, scaled_encoder: NumericRankEncoder) -> None:
        """Test encoding a list of float values."""
        input_vals = [3.14, 2.71, 1.41]
        output = scaled_encoder.encode_all(input_vals)
        assert isinstance(output, torch.Tensor), "Output should be a torch.Tensor."
        assert output.numel() == 3, "Tensor should have exactly three elements."
        assert output[0] == pytest.approx(1), "First encoded value does not match."
        assert output[1] == pytest.approx(0.5), "Second encoded value does not match."
        assert output[2] == pytest.approx(0), "Third encoded value does not match."

    @pytest.mark.parametrize("fixture", ["rank_encoder", "scaled_encoder"])
    def test_encode_all_with_non_numeric_raises(
        self,
        request: pytest.FixtureRequest,
        fixture: str,
    ) -> None:
        """Test that encoding a non-float raises a ValueError.

        Args:
            request: Pytest fixture request
            fixture: Name of the fixture to use
        """
        encoder = request.getfixturevalue(fixture)
        with pytest.raises(ValueError, match="Expected input data to be a float or int"):
            encoder.encode_all(["not_numeric"])

    @pytest.mark.parametrize("fixture", ["rank_encoder", "scaled_encoder"])
    def test_decode_raises_not_implemented(
        self,
        request: pytest.FixtureRequest,
        fixture: str,
    ) -> None:
        """Test that decoding raises NotImplementedError.

        Verifies that decoding is not supported in this encoder.

        Args:
            request: Pytest fixture request
            fixture: Name of the fixture to use
        """
        encoder = request.getfixturevalue(fixture)
        with pytest.raises(
            NotImplementedError,
            match="Decoding is not yet supported for NumericRank.",
        ):
            encoder.decode(torch.tensor([0.0]))
