
import os
import shutil

def delete_cache():
    try:
        RUN_CACHE = "runs"
        if os.path.exists(RUN_CACHE):
            shutil.rmtree(RUN_CACHE)
        for filename in os.listdir('.'):
            if filename.startswith("temp"):
                os.remove(filename)
        return True
    except OSError as e:
        return False
