from flask import Flask, request, render_template
from sklearn.cluster import KMeans
import numpy as np

app = Flask(__name__)

# Simulate training data for demonstration (age, spending_score)
training_data = np.array([[25, 80], [30, 60], [35, 90], [40, 50], [23, 65]])
# Apply K-Means Clustering on training data
kmeans = KMeans(n_clusters=3)
kmeans.fit(training_data)

@app.route('/', methods=['GET', 'POST'])
def index():
    cluster = None
    if request.method == 'POST':
        # Get data from form
        age = float(request.form.get('age'))
        spending_score = float(request.form.get('spending_score'))
        
        # Prepare data for clustering
        data = np.array([[age, spending_score]])
        
        # Predict the cluster for the new data point
        cluster = kmeans.predict(data)[0]
        
    return render_template('index.html', cluster=cluster)

if __name__ == '__main__':
    app.run(debug=True)
