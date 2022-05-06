#!/usr/bin/env python
#_*_coding:utf-8_*_

import re, sys, os, platform
import math
pPath = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(pPath)

USAGE = """
USAGE:
	python PAAC.py input.fasta <lambda> <output>
	input.fasta:      the input protein sequence file in fasta format.
	lambda:           the lambda value, integer, defaule: 30
	output:           the encoding file, default: 'encodings.tsv'
"""

def Rvalue(aa1, aa2, AADict, Matrix):
	return sum([(Matrix[i][AADict[aa1]] - Matrix[i][AADict[aa2]]) ** 2 for i in range(len(Matrix))]) / len(Matrix)

def PAAC(fastas, lambdaValue=30, w=0.05, **kw):

	dataFile = re.sub('codes$', '', os.path.split(os.path.realpath(__file__))[0]) + r'/PAAC.txt' if platform.system() == 'Windows' else re.sub('codes$', '', os.path.split(os.path.realpath(__file__))[0]) + '\PAAC.txt'
	with open(dataFile) as f:
		records = f.readlines()
	AA = ''.join(records[0].rstrip().split()[1:])
	AADict = {}
	for i in range(len(AA)):
		AADict[AA[i]] = i
	AAProperty = []
	AAPropertyNames = []
	for i in range(1, len(records)):
		array = records[i].rstrip().split() if records[i].rstrip() != '' else None
		AAProperty.append([float(j) for j in array[1:]])
		AAPropertyNames.append(array[0])

	AAProperty1 = []
	for i in AAProperty:
		meanI = sum(i) / 20
		fenmu = math.sqrt(sum([(j-meanI)**2 for j in i])/20)
		AAProperty1.append([(j-meanI)/fenmu for j in i])

	encodings = []
	header = ['#']
	for aa in AA:
		header.append('Xc1.' + aa)
	for n in range(1, lambdaValue + 1):
		header.append('Xc2.lambda' + str(n))
	encodings.append(header)

	for i in fastas:
		name, sequence = i[0], re.sub('-', '', i[1])
		code = [name]
		theta = []
		for n in range(1, lambdaValue + 1):
			theta.append(
				sum([Rvalue(sequence[j], sequence[j + n], AADict, AAProperty1) for j in range(len(sequence) - n)]) / (
				len(sequence) - n))
		myDict = {}
		for aa in AA:
			myDict[aa] = sequence.count(aa)
		code = code + [myDict[aa] / (1 + w * sum(theta)) for aa in AA]
		code = code + [(w * j) / (1 + w * sum(theta)) for j in theta]
		encodings.append(code)
	return encodings

if __name__ == '__main__':
	sys.argv[0] ='Train set_532_532.txt'
	from _readFasta import readFasta
	fastas = readFasta(sys.argv[0])
	lambdaValue = 3#int(sys.argv[2]) if len(sys.argv) >= 3 else 30
	output = sys.argv[3] if len(sys.argv) >= 4 else 'encoding.tsv'
	encodings = PAAC(fastas, lambdaValue)
	print('Feature writing to disk is started....')
	with open('AmpPseAAC_Drug.csv','w') as fin:
		for each in encodings:
			fin.write(str(each))
			fin.write('\t\n\t')
	print("Finished")
