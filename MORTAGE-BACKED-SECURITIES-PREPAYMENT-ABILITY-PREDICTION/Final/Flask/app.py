from flask import Flask, render_template, request, url_for, redirect
import pickle
import numpy

app = Flask(__name__)

model=pickle.load(open('model.pkl','rb+'))
scale = pickle.load(open("scaler.pkl",'rb+'))

@app.route('/')
def main():
    return render_template('form.html')

@app.route('/predict',methods=['POST'])
def predict():
        cs = request.form.get("CreditScore")
        fthb = request.form.get("FTHB")
        MSA = request.form.get("MSA")
        MIP = request.form.get("MIP")
        Units = request.form.get("Units")
        Occupancy = request.form.get("Occupancy")
        DTI = request.form.get("DTI")
        OrigUPB = request.form.get("OrigUPB")
        LTV = request.form.get("LTV")
        OrigInterestRate = request.form.get("OrigInterestRate")
        Channel = request.form.get("Channel")
        PropertyType = request.form.get("PropertyType")
        LoanPurpose = request.form.get("LoanPurpose")
        OrigLoanTerm = request.form.get("OrigLoanTerm")
        NumBorrowers = request.form.get("NumBorrowers")
        EverDelinquent = request.form.get("EverDelinquent")
        MonthsInRepayment = request.form.get("MonthsInRepayment")
        

        final_features = [cs, fthb, MSA, MIP, Units, Occupancy, DTI, OrigUPB, LTV, OrigInterestRate, Channel, PropertyType, LoanPurpose, OrigLoanTerm, NumBorrowers, EverDelinquent, MonthsInRepayment ]
        array = numpy.array(final_features)
        scaled_feature = scale.transform(array.reshape(1, -1))
        y_pred = model.predict(scaled_feature)

        if y_pred == 1:
            return render_template('form.html', prediction_text='High chance of prepayment')
        else:
            return render_template('form.html', prediction_text='Low chance of prepayment')

        #return render_template('form.html', prediction_text='Prepayment :> {}'.format(y_pred))

if __name__ == '__main__':
    app.run()
