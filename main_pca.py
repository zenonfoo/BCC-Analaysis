### Principal Component Analysis ###
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from Algorithms.neural_network import training as training

# Loading Data if data
print('Loading Data')
folder_name = 'BCC&NoBCC_Classification/4/BCC_Data_4.npy'
data,X,y = training.import_data(folder_name)

# Feature Scaling
print('Normalizing Data')
sc = StandardScaler()
X = sc.fit_transform(X)

# Performing PCA
pca = PCA()
pca.fit(X)

# Plotting Explained Variance Ratio
plt.bar(range(1,10),pca.explained_variance_ratio_[:9],alpha=0.7)
plt.xlabel('Principal Component Index')
plt.ylabel('Explained Variance Ratio')
plt.legend()

# Plotting First Component
plt.plot(pca.components_[:,0],label='First Principal Component')
plt.plot(pca.components_[:,1],label='Second Principal Component')
plt.plot(pca.components_[:,2],label='Thrid Principal Component')
plt.grid()
plt.title('Principal Componenets When Applying PCA To The Raman Data')
plt.legend()

# Plotting components
for count in range(1,10):

    plt.subplot(3,3,count)
    plt.plot(pca.components_[count,:])
    plt.title('Principal Component ' + str(count))
    plt.grid()