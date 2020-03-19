from run_kitti import *

def make_kitti_video():
     
    basedir = '/media/ubuntu/08382345382330DC/迅雷下载/raw_data'
    date = '2011_09_26'
    drive = '0001'
    dataset = pykitti.raw(basedir, date, drive)
   
    videoname = "detection_{}_{}.avi".format(date, drive)
    save_path = os.path.join(basedir, date, "{}_drive_{}_sync".format(date, drive), videoname)    
    run(dataset, save_path)

make_kitti_video()