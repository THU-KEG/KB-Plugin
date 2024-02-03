 # -*- coding: utf-8 -*-
from utils.soay_api import *
import argparse

parser = argparse.ArgumentParser(description='SoAy Main')

parser.add_argument('--gpt_version', type = str, default='gpt-4', help = 'the same as the version string of OpenAI GPT series')
parser.add_argument('--openai_key', type = str, help = 'openai key from https://www.openai.com/')

args = parser.parse_args()
# openai_api = OpenAI(gpt_version = args.gpt_version, openai_key = args.openai_key)
aminer_api = aminer_soay()

def aminer_searchPublication():
    result = aminer_api.searchPublication(publication_info = 'Incremental Predictive Coding: A Parallel and Fully Automatic Learning\n  Algorithm')
    print(result)
    return result

def aminer_getPublication():
    result = aminer_api.getPublication(pub_id = '5eccb534e06a4c1b26a83738')
    print(result)
    return result

def aminer_getCoauthors():
    result = aminer_api.getCoauthors(person_id = '563455ec45cedb339afabbe4')
    print(result)
    return result

def aminer_searchPersonComp():
    result = aminer_api.searchPersonComp(
        organization = 'Institute of Computing Technology, Chinese Academy of Sciences',
        # interest = "Sample Complexity"
        # name = "Alessandro Epasto"
        )
    print(result)
    print(len(result))
    return result

def aminer_getPersonBasicInfo():
    result = aminer_api.getPersonBasicInfo(person_id='562dd39145cedb3398f1b465')
    print(result)
    return result

def aminer_getPersonInterest():
    getPersonInterest = aminer_api.getPersonInterest
    result = getPersonInterest(person_id='562cbcd845cedb3398cab775')
    print(result)
    return result

def aminer_getPersonPubs():
    result = aminer_api.getPersonPubs(person_id='5d415be17390bff0db70a5ad')
    print(result)
    print(len(result))
    return result

# def chatgpt():
#     response = openai_api.generate_response_chatgpt(query = 'hi')
#     result = response['choices'][0]['message']['content']
#     return result
    # print(response)
    # print()

# api_list = [aminer_searchPersonComp, aminer_searchPublication, aminer_getPublication, aminer_getPersonBasicInfo, aminer_getPersonInterest, aminer_getPersonPubs]
# api_list = [aminer_getPersonPubs]
api_list = [aminer_searchPersonComp]
# api_list = [aminer_getPersonPubs]

for each in api_list:
    try:
        each()
        print('{} is working'.format(each.__name__))
    except:
        print('something wrong with {}, pleaes check'.format(each.__name__))
