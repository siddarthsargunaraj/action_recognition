class Path(object):
    @staticmethod
    def db_dir():
            root_dir = 'data/Original_video'
            output_dir = 'data/output'
            return root_dir, output_dir
            
    @staticmethod
    def model_dir():
        return 'models/c3d-init.pth'
