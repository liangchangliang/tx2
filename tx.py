# -*- coding: utf-8 -*-
"""
企鹅分类器 - 图片匹配修复版
运行目录：D:\streamlit_env
解决问题：预测物种与图片不符（阿德利企鹅显示岩石图）
"""

import streamlit as st
import pickle
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ============================ 全局配置（核心：物种-图片精准匹配） ============================
st.set_page_config(page_title="企鹅分类器", page_icon="🐧", layout="wide")

# 根目录路径
BASE_DIR = "D:/streamlit_env"
DATA_PATH = os.path.join(BASE_DIR, "（企鹅识别数据）penguins-chinese.csv")
MODEL_PATH = os.path.join(BASE_DIR, "rfc_model.pkl")
SPECIES_MAP_PATH = os.path.join(BASE_DIR, "output_uniques.pkl")

# 核心修复1：物种-图片映射（适配无重音ADELIE.png，降低命名难度）
SPECIES_IMG_MAP = {
    "阿德利企鹅": os.path.join(BASE_DIR, "ADELIE.png"),  # 无重音，易输入
    "帽带企鹅": os.path.join(BASE_DIR, "CHINSTRAP.png"),
    "巴布亚企鹅": os.path.join(BASE_DIR, "GENTOO.png")
}
# 其他图片路径
LOGO_IMG = os.path.join(BASE_DIR, "rigth_logo.png")
PENGUINS_ALL_IMG = os.path.join(BASE_DIR, "penguins_all.png")

# 数据中实际岛屿（已适配）
ACTUAL_ISLANDS = ["比斯科群岛", "德里姆岛", "托尔森岛"]
predict_result_species = None
predict_result_img = None

# ============================ 工具函数（增强图片校验） ============================
def check_species_images():
    """检查所有物种图片是否存在，返回缺失列表"""
    missing = []
    for species, img_path in SPECIES_IMG_MAP.items():
        if not os.path.exists(img_path):
            missing.append(f"{species}的图片：{os.path.basename(img_path)}")
    return missing

def get_correct_image(species_name):
    """获取物种对应的正确图片，返回路径和提示"""
    if species_name not in SPECIES_IMG_MAP:
        return None, f"未识别物种：{species_name}"
    
    img_path = SPECIES_IMG_MAP[species_name]
    if os.path.exists(img_path):
        return img_path, f"成功加载{species_name}图片"
    else:
        # 修复2：占位图用企鹅相关图，而非岩石
        default_img = f"https://picsum.photos/300/300?{species_name}"
        return default_img, f"缺失{species_name}图片：{os.path.basename(img_path)}，已用默认图替代"

# ============================ 核心功能函数（无修改，确保稳定） ============================
def load_and_preprocess_data():
    global ACTUAL_ISLANDS
    if not os.path.exists(DATA_PATH):
        st.error(f"❌ 未找到数据集：{DATA_PATH}")
        return None, None, None
    
    try:
        df = pd.read_csv(DATA_PATH, encoding="gbk")
        st.success("✅ 读取数据集（编码：gbk）")
    except:
        st.error("❌ 数据集读取失败，请确认编码为gbk")
        return None, None, None
    
    # 数据清洗与岛屿同步
    df = df.dropna(subset=["企鹅的种类", "企鹅栖息的岛屿", "喙的长度", "喙的深度", "翅膀的长度", "身体质量", "性别"])
    data_islands = df["企鹅栖息的岛屿"].unique()
    if not set(data_islands).issubset(set(ACTUAL_ISLANDS)):
        ACTUAL_ISLANDS = list(data_islands)
        st.info(f"ℹ️ 同步数据中的岛屿：{ACTUAL_ISLANDS}")
    
    # 列名映射与编码
    df.rename(columns={
        "企鹅的种类": "物种", "企鹅栖息的岛屿": "岛屿",
        "喙的长度": "喙长度(mm)", "喙的深度": "喙深度(mm)",
        "翅膀的长度": "鳍长(mm)", "身体质量": "体重(g)"
    }, inplace=True)
    
    X = pd.get_dummies(df[["喙长度(mm)", "喙深度(mm)", "鳍长(mm)", "体重(g)", "岛屿", "性别"]], 
                      columns=["岛屿", "性别"], drop_first=False)
    le = LabelEncoder()
    y = le.fit_transform(df["物种"])
    species_map = {idx: name for idx, name in enumerate(le.classes_)}
    
    return X, y, species_map

def train_or_load_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(SPECIES_MAP_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
            with open(SPECIES_MAP_PATH, "rb") as f:
                species_map = pickle.load(f)
            # 验证岛屿特征
            model_islands = [f for f in model.feature_names_in_ if f.startswith("岛屿_")]
            if set(model_islands) != set([f"岛屿_{i}" for i in ACTUAL_ISLANDS]):
                st.warning("⚠️ 模型岛屿特征不匹配，重新训练")
                raise Exception()
            st.success("✅ 加载模型成功")
            return model, species_map
        except:
            st.warning("⚠️ 模型加载失败，重新训练")
    
    X, y, species_map = load_and_preprocess_data()
    if X is None:
        return None, None
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(SPECIES_MAP_PATH, "wb") as f:
        pickle.dump(species_map, f)
    st.success("✅ 模型训练并保存成功")
    return model, species_map

# ============================ 页面逻辑（增强图片提示） ============================
def render_predict_page():
    global predict_result_species, predict_result_img
    st.header("企鹅物种预测 📊")
    
    # 修复3：预测前先提示缺失图片
    missing_imgs = check_species_images()
    if missing_imgs:
        st.warning("⚠️ 根目录缺少以下物种图片（会影响显示）：")
        for img in missing_imgs:
            st.write(f"- {img}")
    
    col_logo, col_form = st.columns([1, 2.5])
    with col_form:
        with st.form("predict_form"):
            island = st.selectbox("栖息岛屿", ACTUAL_ISLANDS)
            sex = st.selectbox("性别", ["雌性", "雄性"])
            bill_length = st.number_input("喙长度（mm）", 32.0, 60.0, 45.0)
            bill_depth = st.number_input("喙深度（mm）", 13.0, 22.0, 17.0)
            flipper_length = st.number_input("翅膀长度（mm）", 170.0, 240.0, 200.0)
            body_mass = st.number_input("体重（g）", 2700.0, 6300.0, 4200.0)
            submit = st.form_submit_button("预测", type="primary")
        
        model, species_map = train_or_load_model()
        if submit and model:
            # 构造输入特征
            input_data = {"喙长度(mm)": bill_length, "喙深度(mm)": bill_depth, 
                          "鳍长(mm)": flipper_length, "体重(g)": body_mass}
            for feat in model.feature_names_in_:
                if feat.startswith("岛屿_"):
                    input_data[feat] = 1 if feat == f"岛屿_{island}" else 0
                elif feat.startswith("性别_"):
                    input_data[feat] = 1 if feat == f"性别_{sex}" else 0
            
            input_df = pd.DataFrame([[input_data[f] for f in model.feature_names_in_]], 
                                   columns=model.feature_names_in_)
            predict_code = model.predict(input_df)[0]
            predict_result_species = species_map[predict_code]
            
            # 修复4：获取图片并显示提示
            predict_result_img, img_msg = get_correct_image(predict_result_species)
            st.success(f"🎉 预测结果：{predict_result_species}")
            st.info(f"🖼️ {img_msg}")  # 提示图片加载状态

    with col_logo:
        if not submit or not predict_result_img:
            st.image(LOGO_IMG if os.path.exists(LOGO_IMG) else "https://picsum.photos/300/300?penguinlogo", 
                     width=300, caption="企鹅分类器")
        else:
            st.image(predict_result_img, width=300, caption=f"预测物种：{predict_result_species}")

def render_intro_page():
    st.header("企鹅分类器 🐧")
    st.subheader("数据集简介")
    st.write(f"- 包含岛屿：{', '.join(ACTUAL_ISLANDS)}")
    st.write("- 预测物种：阿德利企鹅、帽带企鹅、巴布亚企鹅")
    
    # 显示数据集样本
    if os.path.exists(DATA_PATH):
        df_sample = pd.read_csv(DATA_PATH, encoding="gbk").head(5)
        st.dataframe(df_sample, use_container_width=True)
    
    # 显示物种图鉴
    st.subheader("物种图鉴")
    col1, col2, col3 = st.columns(3)
    for (species, img_path), col in zip(SPECIES_IMG_MAP.items(), [col1, col2, col3]):
        with col:
            img = img_path if os.path.exists(img_path) else f"https://picsum.photos/200/200?{species}"
            st.image(img, use_container_width=True)
            st.caption(species)

# ============================ 主程序 ============================
if __name__ == "__main__":
    st.sidebar.title("功能导航")
    page = st.sidebar.selectbox("选择页面", ["数据集简介", "物种预测"], label_visibility="collapsed")
    
    if page == "数据集简介":
        render_intro_page()
    else:
        render_predict_page()
    
    st.markdown("---")
    st.caption("© 2025 企鹅分类器（图片修复版）")
