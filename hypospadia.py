import streamlit as st  
import pandas as pd  
import pickle  
import matplotlib.pyplot as plt  
import numpy as np  
from sklearn.preprocessing import StandardScaler   
import shap

# Set page title and icon  
st.set_page_config(page_title="DTC Prediction for postoperative complication of UPT", page_icon=":bar_chart:")  

# Add a title and subheader  
st.title("DTC Prediction for postoperative complication of UPT")  
st.subheader("Enter patient information and click Submit to get the prediction")  

# Create a sidebar for input fields  
with st.sidebar:  
    st.header("Input Information")  
    Age = st.number_input("Age (months)")  
    Length_of_penis = st.number_input("Length of penis (cm)")  
    length_of_glans_ = st.number_input("Length of glans (mm)")  
    width_of_glans = st.number_input("Width of glans (mm)")  
    preoperative_curvature = st.number_input("Preoperative curvature (°)")  
    length_of_dificient_urethra = st.number_input("Length of deficient urethra (cm)")  
    Position_of_meatus_before_VC_correction = st.selectbox("Position of meatus before VC correction",
                                                              ['normal', 'fossa navicular', 'coronal sulcus',   
                                                              'distal shaft', 'middle shaft', 'proximal shaft',   
                                                              'penoscrotal', 'scrotum', 'perineum'])  
    submit_button = st.button("Submit")  

# If button is pressed  
if submit_button:  
    try:  
        # Load the saved model from the file  
        with open(r'./DTC.pkl', 'rb') as f:
            clf = pickle.load(f)  

        # Store inputs into dataframe  
        X = pd.DataFrame([[Age, Length_of_penis, length_of_glans_, width_of_glans,  
                           preoperative_curvature, length_of_dificient_urethra,  
                           Position_of_meatus_before_VC_correction]],   
                         columns=['Age', 'Length_of_penis', 'length_of_glans_', 'width_of_glans',  
                                  'preoperative_curvature', 'length_of_dificient_urethra',  
                                  'Position_of_meatus_before_VC_correction'])  

        # 选择需要进行标准化的列  
        columns_to_scale = ['Age', 'Length_of_penis', 'length_of_glans_', 'width_of_glans',  
                    'preoperative_curvature', 'length_of_dificient_urethra']  

        # 实例化 StandardScaler  
                # Load the complete dataset for SHAP analysis  
        BT_complete = pd.read_csv(r'./BT_complete.csv')
        BT= pd.read_csv(r'./UPT_BT.csv')
        c = BT_complete[['Age', 'Length_of_penis', 'length_of_glans_', 'width_of_glans',  
                         'preoperative_curvature', 'length_of_dificient_urethra',  
                         'Position_of_meatus_before_VC_correction']]  
        b=BT[['Age', 'Length_of_penis', 'length_of_glans_', 'width_of_glans',  
                         'preoperative_curvature', 'length_of_dificient_urethra'  
                         ]] 
        
        scaler = StandardScaler()


        # 拟合 StandardScaler 并转换整个数据集
        scaled_df = scaler.fit(b)

        # 假设您有一个单个样本，需要进行标准化
        # 这里我们使用 df 的最后一行作为示例单个样本
        single_sample = X[columns_to_scale] # 将单个样本转换为适合 transform 的格式

        # 使用之前拟合的 StandardScaler 对单个样本进行标准化
        scaled_single_sample = scaler.transform(single_sample)

        # 将标准化后的单个样本转换为 DataFrame 格式
        X1 = pd.DataFrame(scaled_single_sample, columns= columns_to_scale)
        
        mapping = {
         'normal': 1,
         'fossa navicular': 2,
         'coronal sulcus': 3,
         'distal shaft': 4,
         'middle shaft': 5,
         'proximal shaft': 6,
         'penoscrotal': 7,
         'scrotum': 8,
         'perineum': 9
        }
        X1['Position_of_meatus_before_VC_correction'] = X[
            'Position_of_meatus_before_VC_correction'].map(mapping)
        
        
        # Generate predictions  
        prediction = clf.predict(X1)  

        # Initialize SHAP explainer  
        dt_explainer_df = shap.KernelExplainer(clf.predict, c)  
        
        # Calculate SHAP values  
        shap_values_df = dt_explainer_df.shap_values(X1.values)


        # Plot SHAP summary and display it in Streamlit  
        st.subheader("Prediction Result")  
        st.caption("'1' means occurrence of postsurgery complications, '0' means no occurrence of posturgery complications")
        st.write(f"Predicted Result: {prediction[0]}")  # Display prediction  

        
        st.divider()
    
        # Create a waterfall plot for the first prediction  
        st.subheader("SHAP Value Plot")


        #Create SHAP waterfall plot
        expl = shap.Explanation(values=shap_values_df[0], base_values=dt_explainer_df.expected_value,
                                data=X1.iloc[0])
        shap.waterfall_plot(expl, max_display=10)
        plt.title("SHAP Waterfall Plot")  
        
        # Display the plot in Streamlit  
        st.pyplot(plt)
        plt.clf()

    except Exception as e:  
        st.error(f"Error occurred: {e}")
