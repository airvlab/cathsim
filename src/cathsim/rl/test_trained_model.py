from pathlib import Path
import cv2
# from utils import load_sb3_model
from real_env import real_env
config_name: str = "test"

path = Path.cwd() / "15-07-2024/phantom3/bca/pixels/models/sac_0.zip"

# model=load_sb3_model(path, config_name)
myreal_env = real_env()
obs, _ = myreal_env.reset()
# action=model.predict(obs)
# print(action)
# cv2.imshow('Image', obs)
cv2.imshow("filename", obs)
cv2.waitKey(0)
cv2.destroyAllWindows()
