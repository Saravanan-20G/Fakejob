# Job Posting Fraud Detection

This project aims to detect fraudulent job postings using machine learning techniques. It uses a combination of text data from job descriptions and categorical data to build a predictive model.

## Project Structure

job_posting_fraud_detection/
├── job.py # Streamlit app code
├── model_training.py # Model training code
├── requirements.txt # Python dependencies
├── README.md # Project documentation
├── random_forest_model.pkl # Trained model
├── vectorizer.pkl # Fitted TfidfVectorizer
├── feature_names.pkl # List of feature names used during training
├── job_predict.csv # Dataset for dropdown values in Streamlit app

## Usage
Fill in the job posting details such as Job Title, Company Profile, Job Description, Job Requirements, and Job Benefits.
Select the appropriate options for Employment Type, Required Experience, Required Education, Industry, and Function.
Click the "Predict" button to see if the job posting is predicted to be fake or real.
## Dependencies
The project requires the following Python packages:

pandas
scikit-learn
scipy
streamlit
joblib

## Conclusion

The Job Posting Fraud Detection project leverages machine learning to identify potentially fraudulent job postings based on both textual and categorical features. By combining the power of text vectorization with traditional machine learning techniques, this project provides a practical solution for job seekers and employers alike to assess the authenticity of job listings.

With the integration of a Streamlit app, users can easily interact with the model through a user-friendly interface, making predictions in real time. The model's performance can be further enhanced by fine-tuning and incorporating additional features, and the approach demonstrated here serves as a foundation for more advanced fraud detection systems.

We encourage you to explore and experiment with the code, improve the model, and contribute to the development of more robust and effective fraud detection tools. Thank you for using and contributing to this project!

## Future Work

There are several opportunities for enhancing this project, including:

- **Model Improvement:** Experimenting with other machine learning algorithms or neural network architectures to improve prediction accuracy.
- **Feature Expansion:** Including additional features or data sources, such as user reviews or historical job posting data, to increase the model's robustness.
- **Scalability:** Deploying the model in a cloud environment for larger-scale applications and integrating it with job board platforms.
- **User Feedback Integration:** Incorporating user feedback to continuously improve the model's performance and user experience.

We look forward to seeing how this project evolves and hope it serves as a valuable tool in detecting job posting fraud.

Happy coding!
