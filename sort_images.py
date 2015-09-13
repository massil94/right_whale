import os
import pandas as pd

train = pd.read_csv('data/train.csv', index_col='Image')

whaleIDs = list(train['whaleID'].unique())

for w in whaleIDs:
    os.makedirs('./data/imgs/'+w)

for image in train.index:
	folder = train.loc[image, 'whaleID']
	old = './data/imgs/{}'.format(image)        
	new = './data/imgs/{}/{}'.format(folder, image)
	try:
		os.rename(old, new)
	except:
		print('{} - {}'.format(image,folder))