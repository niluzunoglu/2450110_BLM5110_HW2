from sklearn.datasets import make_moons
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

def generate_dataset():
    
    X, y = make_moons(n_samples=400, noise=0.1, random_state=42)

    # Display dataset shape and types
    print(f"Input shape: {X.shape}")
    print(f"Output shape: {y.shape}")

    # Show first few rows of the dataset
    print(f"First few rows of inputs:\n{X[:5]}")
    print(f"First few target values:\n{y[:5]}")

    # Plot the dataset
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('make_moons dataset')
    plt.show()
    
    return X,y

def split_dataset(X,y):
    
    # Train - Test - Validation için 2 kez train_test_split metodunu çağıracağım.
    X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.2, random_state=1)

    
    X_train, X_val, y_train, y_val  = train_test_split(X_train, y_train, test_size=0.25, random_state=1) 
    
    

generate_dataset()