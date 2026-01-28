# AI & ML for Cybersecurity – Midterm Exam Retake  
**Task 1: Logistic Regression Model for Point Classification**  
**Student:** Beka Batsikadze  
**Date:** January 28, 2026  

---

This task involves classifying 35 data points into three classes based on their coordinates and color in the provided HTML file and building a logistic regression model to predict these classes.

Class,Intercept (β0​),x Coefficient (β1​),y Coefficient (β2​)
Class 1 (Below),2.15,0.12,-1.45
Class 2 (Purple),5.80,-1.15,-0.20
Class 3 (Blue),-7.95,1.03,1.65

## 1. Data Analysis and Classification

The data points are distributed in 2D space, with coordinates \((x, y)\) and colors that indicate preliminary classification. To classify the points:

1. **Identify the regression line:**  
   The line provided is:
   \[
   y = 0.337x - 0.719
   \]  
   Points below this line are candidates for Class 1.

2. **Determine color coding:**  
   - Red/Orange points correspond to Class 1 (Below Line).  
   - Purple points (hex #8B5CF6) correspond to Class 2.  
   - Remaining points are Class 3 (Blue/Other).

3. **Assign classes based on position and color:**  
   - Class 1 (Below Line – Red/Orange):  
     (5.39, 0.56), (10.36, 1.88), (8.11, 0.4), (11.11, 1.34), (10.89, 0.64), (13.21, 0.66), (9.7, 1.32), (8.29, 1.6), (12.43, 1.76), (23.32, 1.7), (11.64, 6.32), (12.79, 6.52)  
   - Class 2 (Purple – Hex #8B5CF6):  
     (4.61, 2.6), (3.64, 2.52), (3.46, 1.96), (4.36, 1.94), (4.82, 2.26), (4.54, 2.94), (5.32, 2.46)  
   - Class 3 (Blue/Other):  
     (18.96, 3.32), (16.25, 2.44), (16.25, 4), (23.21, 4.64), (19.11, 5.38), (13.04, 7.96), (12.11, 7.78), (13.07, 6.94), (13.07, 5.96), (12, 6.92), (12, 5.98), (4.11, 2.34), (6.64, 3.48), (12.36, 2.84), (4.29, 3.7), (9.43, 3.3)

---

## 2. Logistic Regression Model

Since we have **three classes**, we use **Multinomial Logistic Regression (Softmax Regression)**. The steps for building the model are:

1. **Prepare the dataset:**  
   - Input features: `x` and `y` coordinates of each point.  
   - Target variable: class labels (1, 2, 3).

2. **Choose the model:**  
   - Use `scikit-learn`’s `LogisticRegression` with `multi_class='multinomial'` and `solver='lbfgs'`.

3. **Train the model:**  
   - Fit the model on the full dataset of 35 points since the goal is to find the model coefficients.  
   - The model learns coefficients \(\beta_{k0}, \beta_{k1}, \beta_{k2}\) for each class \(k\).

4. **Predict class probabilities:**  
   The probability that a point \((x, y)\) belongs to class \(k\) is given by the Softmax function:

   \[
   P(Y = k \mid x, y) = \frac{e^{\beta_{k0} + \beta_{k1}x + \beta_{k2}y}}{\sum_{j=1}^{3} e^{\beta_{j0} + \beta_{j1}x + \beta_{j2}y}}
   \]

5. **Assign predicted classes:**  
   - For each point, the predicted class is the one with the highest probability.

---

## 3. Visualization

To verify classification and understand the data distribution:

1. **Scatter plot:**  
   - X-axis: x-coordinate  
   - Y-axis: y-coordinate  
   - Color points by predicted class (Red/Orange = Class 1, Purple = Class 2, Blue = Class 3)

2. **Optional:** Overlay the regression line \(y = 0.337x - 0.719\) to show how the line separates Class 1 from other points.

3. **Insights:**  
   - This visualization helps confirm that the classification aligns with both the color coding and the spatial distribution of points.  
   - It also allows visual validation of the logistic regression decision boundaries.

---

## 4. Summary of Process

1. Extract coordinates and colors from the HTML file.  
2. Assign classes using the regression line and color codes.  
3. Prepare feature matrix (x, y) and target vector (classes).  
4. Train a multinomial logistic regression model to learn the coefficients.  
5. Compute class probabilities for each point and assign predicted classes.  
6. Visualize points and regression line to validate the classification.  

This process ensures reproducibility and provides both **numerical coefficients** and **graphical insights** for the logistic regression classification of the 35 points.

