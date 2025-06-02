set -e

pip install -r requirements.txt

modelscope download --model BAAI/bge-m3

mkdir db
mkdir BAAI
mv -r bge-m3 BAAI/

python build_db.py