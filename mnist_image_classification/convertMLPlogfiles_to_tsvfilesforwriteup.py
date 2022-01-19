
name = 'mlp50'
file = open(name+'.log', 'r')
lines = file.readlines()
lines = lines[5:-2]
with open(name+'.tsv', 'w') as newfile:
	ind = 0
	while ind<(len(lines)):
		dat = lines[ind:ind+6]
		for i in range(len(dat)):
			dat[i] = dat[i].rstrip('\n')
		print dat[3]
		newfile.write('%s\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\n' 
			%(dat[0], 
				float(dat[3][4:]), 
				100*float(dat[1][19:]), 
				float(dat[4][20:]), 
				100*float(dat[2][16:]), 
				float(dat[5][17:])))
		ind+=6		
file.close()