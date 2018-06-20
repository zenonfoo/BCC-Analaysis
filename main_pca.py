### Principal Component Analysis ###
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from Algorithms.process_data import obtaining_data as obtain
from Algorithms.process_data import data_preprocessing as preprocess
from Algorithms.process_data import baseline_correction as base_correct

from Algorithms.neural_network import training as training

# Loading Data
print('Loading Data')
folder = 'Data/RamanData/tissue_'
label = 'bcc'
label_data,raman_data,tissues_used = obtain.preProcessBCC(folder_name=folder,testing_label=label)
X,shapes = preprocess.organiseData(label_data,raman_data)
X = X[:,:-1]

del label_data
del raman_data

# Baseline Correction
print("Baseline Correction")
X = base_correct.polynomial(X,2)

# Feature Scaling
print('Normalizing Data')
sc = StandardScaler()
X = sc.fit_transform(X)

# Performing PCA
pca = PCA()
pca.fit(X)

# Plotting Explained Variance Ratio
plt.bar(range(1,100),pca.explained_variance_ratio_[:100],alpha=0.7)
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