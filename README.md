# CVE_AI
A small project where an AI model is developed which tries to predict the CVSS score based on the vulnerabilitie's description.  
It is an experiment to find out how high the success rate for the given task is.


### Obtaining the NVD (National Vulnerability Database)

- At the moment there are 19 datasets available (from 2002 until 2021).
- Find the files on the [NVD Website](https://nvd.nist.gov/vuln/data-feeds#JSON_FEED)
- Extract the files into a folder named NVD_DATABASE
```
Example directory structure:

NVD_DATASET
   - nvdcve-1.1-2020.json
   - nvdcve-1.1-2019.json
```

### Classify CVE Security Vulnerabilities with Tensorflow

- The Jupyter Notebook will prepare and train one or multiple datasets in the NVD_DATABASE folder
- The model will be trained with the textual description of each vulnerability and the corresponding severity High, Medium or Low.
- The ```nvd_ai``` and ```nvd_helper``` scripts contain experimental features
