# Why Stimulus?

When I was a PhD student, coming from ML, trying to apply some of the models to bio, I always questioned where the data came from. 

How was the data generated ? does it make sense ? how is the data processed ? which bioinformatic tooling does what ? do the parameters influence the results ? etc. 

Essentially, bio data was always droped in my cloud repo by some mysterious bioinformatician and I had to live with whatever preprocessing was done before I could touch the data. 

Of course, you can always download the raw data, but finding, installing, using the software is a job in itself. 

To solve this, here is the software I always wish I had, `stimulus`. Stimulus allows you to run the grueling bio-data preprocessing and model training in one go. It allows you to remember what was done and share it to for others to reproduce it ! It allows you to use any bioinformatics software out there just by  filling in a config ! It allows you to **measure** the impact of a data transformation step on model performance so that you can confident that you leave no performance on the table. 
