import pandas as pd

def create_feedback_data():
    # 
    feedback_records = {
        "Review_ID": [101, 102, 103, 104, 105],
        "Customer_Name": ["Rahul", "Priya", "Amit", "Sneha", "Vikram"],
        "Feedback_Text": [
            "The product quality is absolutely amazing! Super fast shipping and great support. Highly recommended.",
            "Worst experience ever. The application keeps crashing every time I update it. Terrible service.",
            "Average product. It works fine but the price is too high for these basic features.",
            "Wow, the new update saved me hours of work! The automation tool is pure magic. Thanks!",
            "Disappointed. The item received was broken and customer support is not replying to my emails."
        ]
    }
    
    df = pd.DataFrame(feedback_records)
    df.to_csv("customer_reviews.csv", index=False)
    print("✅ customer_reviews.csv dataset generated successfully!")

if __name__ == "__main__":
    create_feedback_data()
