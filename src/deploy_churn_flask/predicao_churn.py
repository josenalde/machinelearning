from flask import Flask, render_template, request
import pandas as pd
import pickle

# import sklearn

# setar pasta com html e com imagens, css...
app = Flask(__name__, template_folder='template',
            static_folder='template/assets')

# Uma vez treinado o modelo (nuvem etc.) e exportado, pode-se ler para predições
model_rfc = pickle.load(open('./models/pipe.pkl', 'rb'))


@app.route('/')
def home():
    return render_template("homepage.html")


@app.route('/dados_cliente')
def dados_cliente():
    return render_template("form.html")


def get_data():
    tenure = request.form.get('tenure')
    MonthlyCharges = request.form.get('MonthlyCharges')
    TotalCharges = request.form.get('TotalCharges')
    # print('TotalCharge')
    # print(TotalCharges)
    gender = request.form.get('gender')
    SeniorCitizen = request.form.get('SeniorCitizen')
    Partner = request.form.get('Partner')
    Dependents = request.form.get('Dependents')
    PhoneService = request.form.get('PhoneService')
    MultipleLines = request.form.get('MultipleLines')
    InternetService = request.form.get('InternetService')
    OnlineSecurity = request.form.get('OnlineSecurity')
    OnlineBackup = request.form.get('OnlineBackup')
    DeviceProtection = request.form.get('DeviceProtection')
    TechSupport = request.form.get('TechSupport')
    StreamingTV = request.form.get('StreamingTV')
    StreamingMovies = request.form.get('StreamingMovies')
    Contract = request.form.get('Contract')
    PaperlessBilling = request.form.get('PaperlessBilling')
    PaymentMethod = request.form.get('PaymentMethod')

    d_dict = {'tenure': [tenure], 'MonthlyCharges': [MonthlyCharges], 'TotalCharges': [TotalCharges],
              'gender': [gender], 'SeniorCitizen': [SeniorCitizen], 'Partner': [Partner],
              'Dependents': [Dependents], 'PhoneService': [PhoneService],
              'MultipleLines': [MultipleLines], 'InternetService': [InternetService],
              'OnlineSecurity': [OnlineSecurity], 'OnlineBackup': [OnlineBackup],
              'DeviceProtection': [DeviceProtection], 'TechSupport': [TechSupport],
              'StreamingTV': [StreamingTV], 'StreamingMovies': [StreamingMovies],
              'Contract': [Contract], 'PaperlessBilling': [PaperlessBilling],
              'PaymentMethod': [PaymentMethod]}

    return pd.DataFrame.from_dict(d_dict, orient='columns')


@app.route('/send', methods=['POST'])
def show_data():
    df = get_data()
    # colocar dataframe na ordem das features do modelo
    df = df[['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
             'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
             'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
             'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
             'MonthlyCharges', 'TotalCharges']]
    print(df.dtypes)
    df['TotalCharges'] = df['TotalCharges'].astype(float)
    df['MonthlyCharges'] = df['MonthlyCharges'].astype(float)
    df['tenure'] = df['tenure'].astype(float)
    df['SeniorCitizen'] = df['SeniorCitizen'].astype(int)
    print(df.dtypes)
    prediction = model_rfc.predict(df)
    outcome = 'Atenção, cliente potencial de nos deixar...vamos ligar para ele'
    imagem = 'chefe_brabo.jpg'
    if prediction == 0:
        outcome = 'Ufa... esse cliente vai ficar!! Vamos liberar uns filmes para ele...!'
        imagem = 'chefe_feliz.jpg'

    return render_template('result.html', tables=[df.to_html(classes='data', header=True, col_space=10)],
                           result=outcome, imagem=imagem)
    # classes: CSS class(es) to apply to the resulting html table


if __name__ == "__main__":
    app.run(debug=True)
