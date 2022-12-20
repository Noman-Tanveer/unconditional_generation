
# imported the requests library
import os
import requests
import multiprocess
# from multiprocessing import Pool - Note: if you use this, you have to create the process under `if __name__ == '__main__':
from multiprocessing.pool import ThreadPool as Pool
import time


urls = ["https://layoutlm.blob.core.windows.net/docbank/dataset/DocBank_500K_txt.zip",
        "https://layoutlm.blob.core.windows.net/docbank/dataset/MSCOCO_Format_Annotation.zip",
        "https://layoutlm.blob.core.windows.net/docbank/dataset/DocBank_500K_ori_img.zip.001",
        "https://layoutlm.blob.core.windows.net/docbank/dataset/DocBank_500K_ori_img.zip.002",
        "https://layoutlm.blob.core.windows.net/docbank/dataset/DocBank_500K_ori_img.zip.003",
        "https://layoutlm.blob.core.windows.net/docbank/dataset/DocBank_500K_ori_img.zip.004",
        "https://layoutlm.blob.core.windows.net/docbank/dataset/DocBank_500K_ori_img.zip.005",
        "https://layoutlm.blob.core.windows.net/docbank/dataset/DocBank_500K_ori_img.zip.006",
        "https://layoutlm.blob.core.windows.net/docbank/dataset/DocBank_500K_ori_img.zip.007",
        "https://layoutlm.blob.core.windows.net/docbank/dataset/DocBank_500K_ori_img.zip.008",
        "https://layoutlm.blob.core.windows.net/docbank/dataset/DocBank_500K_ori_img.zip.009",
        "https://layoutlm.blob.core.windows.net/docbank/dataset/DocBank_500K_ori_img.zip.010",
        ]

def get_data(url):
    print(f"Downloading {url}")
    # URL of the image to be downloaded is defined as image_url
    r = requests.get(url) # create HTTP response object
    file = os.path.join("../DocBank", url.split("/")[-1])
    # send a HTTP request to the server and save
    # the HTTP response in a response object called r
    with open(file, 'wb') as f:

        # Saving received content as a png file in
        # binary format

        # write the contents of the response (r.content)
        # to a new file in binary mode.
        f.write(r.content)
    return



# if __name__ == "__main__": # uncomment this and add indentation on below code, if using Pool instead of ThreadPool

## with multi-processing
p = Pool()
t = time.time()
res1 = p.map(get_data, urls)
print('With multi-processing', time.time()-t)

t = time.time()
print(f"Time taken for unpacking result: {time.time()-t}")
