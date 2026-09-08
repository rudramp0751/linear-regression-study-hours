import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error



def main(): 
    print("Linear Regression Project Running...")

    # dataset
    data = {
        "Hours": [1, 2, 3, 4, 5, 6, 7, 8],
        "Scores": [50, 55, 60, 65, 70, 75, 80, 85]
    }

    df = pd.DataFrame(data)
    print(df)

    # Scatter plot
    plt.scatter(df["Hours"], df["Scores"], color="blue")
    plt.xlabel("Hours Studied")
    plt.ylabel("Exam Score")
    plt.title("Study Hours vs Exam Score")
    plt.show()

    x = df[["Hours"]]
    y = df["Scores"]

    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    print("Predicted Scores:", y_pred)
    print("Actual Scores:", list(y_test)) 

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print("R² Score:", r2)
    print("Mean Squared Error:", mse)

    #Linear Regression Line
    plt.scatter(x, y, color= 'red')
    plt.plot(x, model.predict(x), color='blue')
    plt.xlabel("Hours Studied")
    plt.ylabel("Exam Score")
    plt.title("Linear Regression Line")
    plt.show()

    #Slope and intercept
    print("Slope (Coefficient):", model.coef_[0])
    print("Intercept:", model.intercept_)

    ##Residual Plot
    residual = y_test - y_pred 

    plt.scatter(x_test, residual)
    plt.axhline(y=0, color = 'black')
    plt.xlabel("Hours Studied")
    plt.ylabel("Residual Value (Actual - Predicted)")
    plt.title("Residual Plot")
    plt.show()

if __name__ == "__main__":
    main()
