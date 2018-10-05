timestamp=`date +%s`
datetime=`date -d @$timestamp +"%Y-%m-%d-%H:%M:%S"`
#net=AlexNet
#net=VGG16
#net=InceptionV3
net=ResNet50
gpu_id=0
THEANO_FLAGS='device=gpu'$gpu_id',floatX=float32,lib.cnmem=0.6' python cub_demo.py $net | tee "./log/"$net"-"$datetime".log.txt"


