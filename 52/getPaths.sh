#! /bin/bash
# get gray PNG images for sampling 

if [ "$#" -ne 1 ]
then
	echo -e "\nI need to know the directory from which images can be sampled.\n"
	echo -e "You should run this script as\n\n\t ./getPaths.sh /path/to/image/directory\n\n"
	echo -e "or \n\n\t./getPaths.sh \$(cat file) \n\nif the directory can be found in file\n"
	exit
else
	echo -e "The directory is \n\n\t$1\n"
fi

echo -e "Only grayscale PNG will be used.\n"

#see if there is already a list.txt
duplicate=$(find . -maxdepth 1 -type f -name "list.txt" | wc -l)
if [ "$duplicate" -ne 0 ]
then 
	read -p "delete the existing list.txt? y/n " answer
	if [[ "$answer" == y ]]
	then
		rm list.txt
		echo ""
	else
		#do nothing
		exit
	fi
fi

#find all the PNG files 
find $1 -type f -name "*.png" -exec sh -c "echo {} >> list" \;
num_images=$(cat list | wc -l)
echo -e "$num_images PNG formatted images found in this folder.\n"

#Only take those grayscale PNG files
cat list | xargs -I name file name | grep grayscale >> grayList
num_images=$(cat grayList | wc -l)
echo -e "$num_images are grayscale images.\n"

#record their paths in list.txt
cat grayList | sed 's/\(^.*\):.*/\1/g' >> list.txt
echo -e "list.txt is created!\n"

#clean up
rm grayList
rm list
