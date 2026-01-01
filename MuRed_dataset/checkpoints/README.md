# Model Checkpoints

This directory contains the trained model checkpoints required to run the project.  
Due to GitHub's 25MB file size limit, the `.pth` model files are stored externally.

---

## 📥 Download Model Files

| File Name             | Description                           | Download Link |
|------------------------|----------------------------------------|----------------|
| `main_checkpoint.pth` | Primary trained model checkpoint       | https://drive.google.com/file/d/1Em02tWmYaD6D2P9BDnDNX1C8yLR1SU0/view?usp=sharing |
| `best_model.pth`      | Best performing model (evaluation)     | https://drive.google.com/file/d/1W3QpyG6E6-UMA29SS_bUCirnK95eL7zY/view?usp=sharing |

---

## 📂 How to Use

After downloading, place the files in this folder with the Checkpoints:
MuRed_dataset/

└── checkpoint/

├── main_checkpoint.pth

├── best_model.pth

└── README.md

---

## ⚙️ Notes

- Make sure both files are present before running inference or training resume.
- If you're using this repo in a new environment, **download the weights first**.
- Do not rename the files unless you update the model loading script accordingly.

---

If the download links break or access is restricted, please update the links in this file.


