# AIC - Machine Learning Strategies Repository

In this repository we organize our machine learning training scripts. 
Some of our models can be obtained in our [HuggingFace Repos](https://huggingface.co/Aachen-Investment-Club). 


## Setup
Create a virtual environment
```sh
python -m venv venv
```

Install dependencies: 
```sh
pip install -r requirements.txt
```



## Setup (portfolio management subrepo)

After cloning the repo, clone our `portfolio-management` in the root dir of this repo. (Ignore the setup describedin the subrepo).

1. Install the requirements: 
```sh
pip install -r /portfolio-management/requirements.txt
```
2. Create a subfolder `/portfolio-management/data`. 

3. Download the necessary setup data from the following URL, and 
place them in the `/portfolio-management/data` folder. (From the root dir of the repo)
- [google drive](https://drive.google.com/drive/folders/1BEqdjOI4otPHc3-4iggC3r244uuY0_z4?usp=sharing)




4. Make sure to first setup the DB using 
```sh
cd ./portfolio-management # if you are not in this folder yet
python -m porfolio
```

5. MAKE SURE TO FIRST SETUP THE DB AS ABOVE. 
Set the following in your `.env` 
```sh
DB_PATH = sqlite:///portfolio-management/market.db
```

