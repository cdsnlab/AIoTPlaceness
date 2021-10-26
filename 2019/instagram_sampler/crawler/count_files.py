import os
root = '.'

img_dir = ["20180301~20180930", "20180931~"]
location_ids = [i for i in os.listdir(".") if os.path.isdir(i)]

location_ids = []
with open("location_ids") as fp:
	for line in fp:
		location_ids.append(line.strip())

tot = 0
posts = 0
count_dict = {}
jpg_list = open("tot_jpg_list.txt", 'w')
json_list = open("tot_json_list.txt", 'w')
for idir in img_dir:
	for loc_id in location_ids:
		if os.path.isdir(idir + '/%'+loc_id):
			count_dict[loc_id] = 0
			for name in os.listdir(idir + '/%'+loc_id):
				if name[-4:] == '.jpg':
					jpg_list.write("{}/{}/{}\n".format(idir, loc_id, name))
					count_dict[loc_id] += 1
					tot += 1
				if name[-5:] == '.json':
					json_list.write("{}/{}/{}\n".format(idir, loc_id, name))
					count_dict[loc_id] += 1
					posts += 1

for k, v in count_dict.items():
	print(k, v)
#print(count_dict)
print(len(count_dict), tot, posts)
