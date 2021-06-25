class Path(object):
    @staticmethod
    def db_dir(database):
        if database == 'Original_video':
            root_dir = 'data/Original_video'

            # Save preprocess data into output_dir
            output_dir = 'data/output'

            return root_dir, output_dir
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError

    @staticmethod
    def model_dir():
        return 'models/c3d-init.pth'
