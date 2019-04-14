"""Confusion Matrix for categorical data

2019 Colin Dietrich
"""

class Download():
    def __init__(self, data_url, data_directory):
        self.data_directory = data_directory
        self.data_url = data_url
        self.data_file = data_url.split("/")[-1]
        self.data_file_target = os.path.normpath(data_directory +
                                                 os.path.sep +
                                                 self.data_file)

    @staticmethod
    def download_data(d_url, d_directory, verbose=False):
        """Download data to local directory

        Parameters
        ----------
        d_directory : str, path to directory to save file
        d_url : str, URL to download file
        """
        if verbose:
                print('Downloading data from %s...' % d_url)
        d_file = d_url.split("/")[-1]
        d_file_target = os.path.normpath(d_directory + os.path.sep + d_file)
        if not os.path.exists(d_file_target):
            res = requests.get(d_url)
            with open(d_file_target, "wb") as f:
                f.write(res.content)
            if verbose:
                print('Data is located in %s' % d_directory)
        else:
            print('Download file already exists in {}'.format(d_directory))

    @staticmethod
    def extract_files(d_file, d_directory=".", verbose=False):
        """Extract tar or tar.gz archive files

        Parameters
        ----------
        d_file : str, file name of archive to extract
        d_directory : str, path to directory to save file
        """
        if d_file.endswith("tar.gz") or d_file.endswith("tgz"):
            tar = tarfile.open(d_file, "r:gz")
            tar.extractall(path=d_directory)
            tar.close()
        elif d_file.endswith("tar") or d_file.endswith("tarfile"):
            tar = tarfile.open(d_file, "r:")
            tar.extractall(path=d_directory)
            tar.close()
        if verbose:
            print("Files extracted successfully to {}".format(d_directory))

    def extract(self, verbose=False):
        self.download_data(self.data_url, self.data_directory, verbose)
        self.extract_files(self.data_file_target, self.data_directory, verbose)

if __name__ == "__main__":
    import os
    directory = os.path.normpath("")
    gs = Download(_,_)
    gs.download_data(d_url="https://upload.wikimedia.org/wikipedia/commons/a/a9/Female_German_Shepherd.jpg",
                 d_directory=directory)
    url = 'http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar'
    d_images = Download(data_url=url, data_directory=directory)
    d_images.extract(verbose=True)