from SPARQLWrapper import SPARQLWrapper, XML
import re
from bs4 import BeautifulSoup
import requests


def _execute_query(query_string):
    sparql = SPARQLWrapper("https://dbpedia.org/sparql")
    sparql.setQuery(query_string)
    sparql.setReturnFormat(XML)
    try:
        qres = sparql.query().convert()
    except:
        return None
    results = qres.getElementsByTagName("uri")
    if len(results) > 0:
        return str(results[0].firstChild.nodeValue)
    return None


def query(element):
    query_string = "SELECT ?s WHERE {{?s rdfs:label \"" + element + "\"@en ;a owl:Thing .}}"
    result = _execute_query(query_string)
    if result is not None:
        return result
    else:
        query_string = "SELECT ?s WHERE {{?altName rdfs:label \"" + element + "\"@en ;dbo:wikiPageRedirects ?s .}}"
        result = _execute_query(query_string)
        if result is not None:
            return result
        else:
            uri = _search_page(element)
            if uri is not None:
                return str(uri)
            else:
                return None
    return None


def _search_page(element):
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Max-Age': '3600',
        'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0'
    }
    url = "https://dbpedia.org/page/Magical_creatures_in_Harry_Potter"
    req = requests.get(url, headers)
    soup = BeautifulSoup(req.content, 'lxml')
    elements = soup.find_all('a')
    exp = ".*:"+element+".*"
    for element in elements:
        if element.find(string=re.compile(exp)):
            return element['href']
    return None

