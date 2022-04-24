import sys,json

f = open(sys.argv[1], 'r') #input.ipynb
j = json.load(f)
of = open(sys.argv[1][:-5]+'py', 'w') #output.py
if j["nbformat"] >=4:
	for i,cell in enumerate(j["cells"]):
		of.write("#%%\n")
		for line in cell["source"]:
			if cell["cell_type"] == "markdown":
				of.write('# '+ line)
			else:
				of.write(line)
		of.write('\n\n')
of.close()