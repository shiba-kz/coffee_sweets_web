import streamlit as st 
import os.path,sys
sys.path.append(os.path.abspath(os.path.dirname(''))+'/../../')
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D,Input
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
import keras.callbacks
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
from PIL import Image
import requests
from bs4 import BeautifulSoup
import pandas as pd
from time import sleep
import requests
from bs4 import BeautifulSoup
from keras.models import load_model
import pickle

st.set_page_config(
    page_title="Sweets&Coffee App",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)

try:
    model = load_model('1115poch5batch8wari75.hdf5')
except Exception as e:
    st.error(f"モデルのロードに失敗しました: {e}")
    st.stop()

sweets_classes_names = {
    0: ["アップルパイ"], 
    1: ["ブルーベリーケーキ"], 
    2: ["カヌレ"], 
    3: ["チョコケーキ"], 
    4: ["チョコチップクッキー"], 
    5: ["プレーンクッキー"], 
    6: ["シュークリーム"], 
    7: ["エクレア"], 
    8: ["フルーツタルト"], 
    9: ["ブドウケーキ"], 
    10: ["レモンパイ"], 
    11: ["ナッツタルト"], 
    12: ["もものケーキ"], 
    13: ["イチゴのショートケーキ"], 
    14: ["ワッフル"]
}

# クラス名の設定
flavor_classes_names = {
    0: ["アップル", "林檎", "リンゴ", "りんご"], 
    1: ["ブルーベリー", "ベリー"], 
    2: ["バニラ"], 
    3: ["チョコ"], 
    4: ["チョコ"], 
    5: ["スイート", "スウィート", "甘味", "甘さ"], 
    6: ["バニラ"], 
    7: ["チョコ", "バニラ"], 
    8: ["フルーティー", "果実"], 
    9: ["ブドウ", "ぶどう", "グレープ", "マスカット"], 
    10: ["レモン", "れもん", "檸檬", "柑橘", "シトラス"], 
    11: ["アーモンド", "ナッツ"], 
    12: ["ピーチ", "もも", "黄桃"], 
    13: ["ストロベリー", "イチゴ", "いちご", "苺", "ベリー"], 
    14: ["蜂蜜", "はちみつ", "ハニー", "メープル", "カラメル"]
}

# Streamlitタイトル
st.title('スイーツに相性がよいコーヒー を提案いたします！')
st.subheader('🍰スイーツの画像をアップロードしてください🍰')
st.caption('※画像のアップロード後にコーヒー豆を提案いたします！')

# 画像アップロード
uploaded_file = st.file_uploader("画像を選択してください...", type='jpg')

if uploaded_file is None:
    st.warning("画像がアップロードされていません。画像をアップロードしてください。")
    st.stop()

if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file)
        st.image(img, caption='アップロードされた画像', use_column_width=True)

        # 画像の前処理
        IMAGE_SIZE = 224
        img_resized = img.resize((IMAGE_SIZE, IMAGE_SIZE))
        x = img_to_array(img_resized)
        x = np.expand_dims(x, axis=0)
        

        # 予測
        predict = model.predict(preprocess_input(x))

        for pre in predict:
            y = pre.argmax()  # 最大の予測値を持つクラスを取得
            flavor_class_name = flavor_classes_names[y] 
            sweets_class_name = sweets_classes_names[y]  # クラス名のリストを取得
            confidence = pre[y]  # そのクラスの確率を取得
            

            # 結果表示
            if confidence >= 0.7:
                st.success(f"予測結果: {sweets_class_name} (確率: {confidence:.2f})")
                keywords = flavor_class_name
            else:
                st.warning("すみません、認識できませんでした。")

        if keywords is None:
            st.error("キーワードが生成されていないため、検索をスキップします。")
            st.stop()

    except Exception as e:
        st.error(f"画像処理中にエラーが発生しました: {e}")
        st.stop()

st.title("☕おすすめのコーヒー豆☕") 
# 各サイトのルートURLと商品一覧ページのURLを定義
root_urls = {
    'wataru': 'https://watarucoffee-onlinestore.com/',
    'hirocoffee': 'https://www.hirocoffee.jp',
    'tonya': 'https://www.tonya.co.jp',
    'tokiwa': 'https://tokiwacoffee.com',
    'toshino': 'http://shop.toshinocoffee.jp/'
}

urls = {
    'wataru': ['https://watarucoffee-onlinestore.com/?mode=cate&cbid=2666252&csid=0'],
    'hirocoffee': ['https://www.hirocoffee.jp/?mode=f1'],
    'tonya': ['https://www.tonya.co.jp/shop/c/c2005'],
    'tokiwa': [
        'https://tokiwacoffee.com/item_cat/beans/',
        'https://tokiwacoffee.com/item_cat/beans/page/2/',
        'https://tokiwacoffee.com/item_cat/beans/page/3/',
        'https://tokiwacoffee.com/item_cat/beans/page/4/',
        'https://tokiwacoffee.com/item_cat/beans/page/5/'
    ],
    'toshino': [
        'http://shop.toshinocoffee.jp/?mode=cate&cbid=1583225&csid=0',
        'http://shop.toshinocoffee.jp/?mode=cate&cbid=1583225&csid=0&page=2'
    ]
}

# 各サイトごとの商品詳細ページのURLを保存するリスト
all_product_urls = {
    'wataru': [],
    'hirocoffee': [],
    'tonya': [],
    'tokiwa': [],
    'toshino': []
}

# 各サイトから商品詳細ページのURLを取得
for store, store_urls in urls.items():
    for url in store_urls:
        res = requests.get(url)
        if res.status_code != 200:
            print(f"Failed to retrieve {url}")
            continue
        
        soup = BeautifulSoup(res.text, 'html.parser')

        if store == 'wataru':
            beans = soup.find('ul', class_='c-product-list')
            if beans:
                product_items = beans.find_all('li', class_='c-product-list__item')
                for item in product_items:
                    product_link_tag = item.find('a')
                    if product_link_tag:
                        bean_url = product_link_tag['href']
                        product_url = root_urls[store] + bean_url
                        all_product_urls[store].append(product_url)
        
        elif store == 'hirocoffee':
            beans = soup.find('div', class_='p-item-sec-wrap')
            if beans:
                product_items = beans.find_all('div', class_='p-item-box')
                for item in product_items:
                    product_link_tag = item.find('a')
                    if product_link_tag:
                        product_url = product_link_tag['href']
                        all_product_urls[store].append(product_url)
        
        elif store == 'tonya':
            beans = soup.find('div', class_='StyleT_Line_ tile_line_')
            if beans:
                product_items = beans.find_all('div', class_='StyleT_Item_ tile_item_ C2005')
                for item in product_items:
                    product_link_tag = item.find('a')
                    if product_link_tag:
                        bean_url = product_link_tag['href']
                        product_url = root_urls[store] + bean_url
                        all_product_urls[store].append(product_url)
        
        elif store == 'tokiwa':
            beans = soup.find('div', id='main')
            if beans:
                product_items = beans.find_all('li')
                for item in product_items:
                    product_link_tag = item.find('a')
                    if product_link_tag:
                        product_url = product_link_tag['href']
                        all_product_urls[store].append(product_url)
        
        elif store == 'toshino':
            beans = soup.find('div', class_='category_items')
            if beans:
                product_items = beans.find_all('div',class_ = 'item_thumbnail')
                for item in product_items:
                    product_link_tag = item.find('a')
                    if product_link_tag:
                        bean_url = product_link_tag['href']
                        product_url = root_urls[store] + bean_url
                        all_product_urls[store].append(product_url)
                        

# 該当する商品を保存するリスト
matching_products = []

# 各商品詳細ページでキーワードを検索
for store, product_urls in all_product_urls.items():
    for product_url in product_urls:
        res = requests.get(product_url)
        if res.status_code != 200:
            #print(f"Failed to retrieve {product_url}")
            continue
        
        soup = BeautifulSoup(res.text, 'html.parser')
        
        if store == 'wataru':
            product_article = soup.find('div', class_='p-product-body')
            if product_article:
                cupping_notes = product_article.find_all('div', class_='m-table')
                for note in cupping_notes:
                    note_text_tag = note.find('tr')
                    if note_text_tag:
                        note_text = note_text_tag.get_text()
                        if any(keyword in note_text for keyword in keywords):
                            product_name_tag = product_article.find('div', class_='p-product-body__name')
                            if product_name_tag:
                                product_name = product_name_tag.get_text(strip=True)
                                matching_products.append({'URL': product_url, '銘柄': product_name})
        
        elif store == 'hirocoffee':
            product_article = soup.find('div', class_='p-product-body')
            if product_article:
                cupping_notes = product_article.find_all('div', class_='p-product-body-inner')
                for note in cupping_notes:
                    note_text_tag = note.find('p')
                    if note_text_tag:
                        note_text = note_text_tag.get_text()
                        if any(keyword in note_text for keyword in keywords):
                            product_name_tag = product_article.find('div', class_='p-product-body__name')
                            if product_name_tag:
                                product_name = product_name_tag.get_text(strip=True)
                                matching_products.append({ 'URL': product_url, '銘柄': product_name})

        elif store == 'tonya':
            product_article = soup.find('div', class_='goodsproductdetail_inner_')
            if product_article:
                cupping_notes = product_article.find_all('div', class_='goodsspec_area_', id='cart_opt')
                for note in cupping_notes:
                    note_text_tag = note.find('div', class_='goodscomment2_')
                    if note_text_tag:
                        note_text = note_text_tag.get_text()
                        if any(keyword in note_text for keyword in keywords):
                            product_name_tag = product_article.find('h2', class_='goods_rifhtname_')
                            if product_name_tag:
                                product_name = product_name_tag.get_text(strip=True)
                                matching_products.append({'URL': product_url, '銘柄': product_name})

        elif store == 'tokiwa':
            product_article = soup.find('article', class_='article_item')
            if product_article:
                cupping_notes = product_article.find_all('dl', class_='dlist_cupping')
                for note in cupping_notes:
                    note_text_tag = note.find('dd')
                    if note_text_tag:
                        note_text = note_text_tag.get_text()
                        if any(keyword in note_text for keyword in keywords):
                            product_name_tag = product_article.find('h1')
                            if product_name_tag:
                                product_name = product_name_tag.get_text(strip=True)
                                matching_products.append({'URL': product_url, '銘柄': product_name})

        elif store == 'toshino':
            product_article = soup.find('tr', valign='top')
            if product_article:
                cupping_notes = product_article.find('div', id='detail')
                if cupping_notes:
                    note_text = cupping_notes.get_text()
                    if any(keyword in note_text for keyword in keywords):
                        product_name_tag = soup.find('p', class_='pagetitle')
                        if product_name_tag:
                            product_name = product_name_tag.get_text(strip=True)
                            matching_products.append({'URL': product_url, '銘柄': product_name})

            # 結果の表示
if matching_products:
    df = pd.DataFrame(matching_products)
    st.table(df)
else:
    st.warning("該当する商品が見つかりませんでした。")





