import os
import json


def load_json_nvd(directory):
    nvd={}
    dir_list=os.listdir(directory)
    for filename in dir_list:
        if '.json' in filename:
            fp=open(F"{directory}/{filename}", 'r', encoding='utf-8')
            nvd[filename]=json.load(fp)
            fp.close()
        print(F'\r{int(dir_list.index(filename)/len(os.listdir(directory))*100)}%', end='', flush=True)
    print()
    return nvd

# extract a vector containing different combinations of data using this function

# properties are generally suitable for the input layer
# while scores and severity rankings are suitable for the output layer
cve_key={
    'base_metric_v2_properties': ['obtainAllPrivilege', 'obtainUserPrivilege', 'obtainOtherPrivilege', 'userInteractionRequired'],
    'cvssv2_properties': ['accessVector', 'accessComplexity', 'authentication', 'confidentialityImpact', 'integrityImpact', 'availabilityImpact'],
    'cvssv2_base_score': 'baseScore',
    'bmv2_severity': 'severity',
    'bmv2_exploitability_score': 'exploitabilityScore',
    'bmv2_impact_score': 'impactScore',
    'cvssv3_properties': ['attackVector', 'attackComplexity', 'privilegesRequired', 'userInteractionRequired', 'scope', 'confidentialityImpact', 'integrityImpact', 'availabilityImpact'],
    'cvssv3_base_score': 'baseScore',
    'cvssv3_base_severity': 'baseSeverity',
    'bmv3_exploitability_score': 'exploitabilityScore',
    'bmv3_impact_score': 'impactScore',
    'description': 'value',
}

def extract_data(nvd, keys_get):
    passed = 0
    failed = 0
    items_vector=[]
    for year in nvd.keys():
        i=0
        for cve_item in nvd[year]['CVE_Items']:
            new_item=()
            i+=1
            for key in keys_get:
                cve_val=[]
                try:
                    if key == 'base_metric_v2_properties':
                        for i in key: cve_val.append(str(cve_item['impact']['baseMetricV2'][cve_key[i]]))
                    elif key == 'cvssv2_properties':
                        for i in key: cve_val.append(str(cve_item['impact']['baseMetricV2'][cve_key[i]]))
                    elif key == 'cvssv2_base_score':
                        cve_val.append(str(cve_item['impact']['baseMetricV2']['cvssV2'][cve_key[key]]))
                    elif key == 'bmv2_severity':
                        cve_val.append(str(cve_item['impact']['baseMetricV2'][cve_key[key]]))
                    elif key == 'bmv2_exploitability_score':
                        cve_val.append(str(cve_item['impact']['baseMetricV2'][cve_key[key]]))
                    elif key == 'bmv2_impact_score':
                        cve_val.append(str(cve_item['impact']['baseMetricV2'][cve_key[key]]))
                    elif key == 'cvssv3_properties':
                        for i in key: cve_val.append(str(cve_item['impact']['baseMetricV2']['cvssV3'][cve_key[i]]))
                    elif key == 'cvssv3_base_score':
                        cve_val.append(str(cve_item['impact']['baseMetricV3']['cvssV3'][cve_key[key]]))
                    elif key == 'cvssv3_base_severity':
                        cve_val.append(str(cve_item['impact']['baseMetricV3']['cvssV3'][cve_key[key]]))
                    elif key == 'bmv3_exploitability_score':
                        cve_val.append(str(cve_item['impact']['baseMetricV3'][cve_key[key]]))
                    elif key == 'bmv3_impact_score':
                        cve_val.append(str(cve_item['impact']['baseMetricV3'][cve_key[key]]))
                    elif key == 'description':
                        cve_val.append(str(cve_item['cve']['description']['description_data'][0][cve_key[key]]))
                        print(F"original {new_item} append {tuple(cve_val)} key {key} keys {keys_get}")
                    new_item=new_item+tuple(cve_val)
                    passed+=1
                except:
                    failed+=1
                    break
            items_vector.append(new_item)
            # if i == 100: return items_vector
    print(F"passed: {passed}\nfailed: {failed}")
    return items_vector


