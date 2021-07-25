import torch
import os
from tqdm import tqdm
import multiprocessing



from functools import partial, lru_cache
from PIL import Image

from shutil import copy2
from facenet_pytorch import MTCNN, InceptionResnetV1, extract_face
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from torch.utils.data import DataLoader
from torchvision import datasets

from torchvision.transforms import ToTensor


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Running on device: {device}')

NUM_CORES = multiprocessing.cpu_count()

def get_face_files_from_dir(dir):
    file_names = [d.name for d in os.scandir(dir)]
    return file_names

@lru_cache
def get_embeddings(face_files):

    margin = 80
    image_size = 160

    # Load face detector
    mtcnn = MTCNN(keep_all=False, select_largest=False, post_process=False,
                device=device, min_face_size=80,
                margin=margin, image_size=image_size).eval()

    # Load facial recognition model
    resnet = InceptionResnetV1(pretrained='vggface2', device=device).eval()

    #Calculuate embeddings
    tf_img = lambda i: ToTensor()(i)
    embeddings = lambda input: resnet(input)
    
    
    normalize = lambda t: (t - torch.min(t)) / (torch.max(t) - torch.min(t))

    list_embs = []
    with torch.no_grad():
        for face in tqdm(face_files):
            # t = mtcnn(t)
            e = embeddings(face).squeeze()
            list_embs.append(e)


    converted_list_embs = [t.cpu().numpy() for t in list_embs]
    return converted_list_embs


def put_files_in_cluster_folders(face_files, clusters):
    assert len(clusters) == len(face_files), "IO lengths should be the same"

    dest_test_folder = "./test_clustering"
    if not os.path.isdir(dest_test_folder):
        os.mkdir(dest_test_folder)

    for i in range(len(face_files)):
        cluster_id = clusters[i]
        file_name = face_files[i]

        cluster_dir = os.path.join(dest_test_folder, str(cluster_id))
        
        if not os.path.isdir(cluster_dir):
            os.mkdir(cluster_dir)

        file_src  = os.path.join(file_name)
        file_dst = os.path.join(cluster_dir, file_name.split('/')[-1])

        copy2(file_src,  file_dst)
    
    print("Done copying")

def run():

    dir = "./data/DeepFashionMenOnlyCleaned/SourceImages"


    face_files = list(map(partial(os.path.join, dir), get_face_files_from_dir(dir)))
    print(len(face_files))

    list_embs = get_embeddings(tuple(face_files))
    print(len(list_embs))


    #Clustering, parallelized through n_jobs
    x=list_embs
    x = PCA(n_components=128).fit_transform(x)
    x = TSNE(perplexity=128, n_components=3).fit_transform(x)

    clusters = DBSCAN(eps=0.50, min_samples=4, n_jobs=NUM_CORES).fit_predict(x)
    print(len(clusters))

    put_files_in_cluster_folders(face_files, clusters)
    print("Finished")


if __name__ == "__main__":
    run()
    # t_list = ["MEN_Shirts_Polos_id_00004343_01_1_front.jpg", "MEN_Sweatshirts_Hoodies_id_00002774_02_1_front.jpg", "MEN_Tees_Tanks_id_00003425_05_1_front.jpg", "MEN_Tees_Tanks_id_00007967_02_4_full.jpg"]
    # put_files_in_cluster_folders(t_list, [1, 2, 1, 5])