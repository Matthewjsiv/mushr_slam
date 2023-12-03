import numpy as np
import math
from sklearn.decomposition import PCA
''' Multiview 2D projection (M2DP) descriptor.

 Input:
       data        n*3     Point cloud. Each row is [x y z]
 Output:
       desM2DP     192*1   M2DP descriptor of the input cloud data
       A           64*128  Signature matrix

 Introduction:
 M2DP is a global descriptor of input point cloud. Details of M2DP can be
 found in the following paper:

 Li He, Xiaolong Wang and Hong Zhang, M2DP: A Novel 3D Point Cloud
 Descriptor and Its Application in Loop Closure Detection, IROS 2016.

 Li He, Dept. of Computing Science, University of Alberta
 lhe2@ualberta.ca
'''

def cart2sph(x,y,z):
    azimuth = np.arctan2(y,x)
    elevation = np.arctan2(z,np.sqrt(x**2 + y**2))
    r = np.sqrt(x**2 + y**2 + z**2)
    return azimuth, elevation, r

def sph2cart(azimuth,elevation,r):
    x = r * np.cos(elevation) * np.cos(azimuth)
    y = r * np.cos(elevation) * np.sin(azimuth)
    z = r * np.sin(elevation)
    return x, y, z
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)
def GetSignatureMatrix(azimuthList,elevationList, cloud_pca,
                       numT, numR, maxRho):
    shape_A=(azimuthList.shape[0]*elevationList.shape[0],numT*numR)
    thetaList=np.linspace(-math.pi,math.pi,numT+1)
    #rho list
    rhoList = np.linspace(0,np.sqrt(maxRho),numR+1)
    rhoList = np.square(rhoList)
    rhoList[-1] = rhoList[-1]+.001 #make sure all points in bins
    n=0
    A=[]
    #loop on azimuth
    for azm in azimuthList:
        #loop on evevation
        for elv in elevationList:
            #normal vector vecN of the selected 2D plane
            x,y,z=sph2cart(azm,elv,1)
            vecN=np.array([x,y,z])
            #distance of vector [1,0,0] to the surface with normal vector vecN
            op= np.array([1,0,0])
            h = op@vecN.T
            #a new vector, c = h*vecN, so that vector [1,0,0]-c is the
            # projection of x-axis onto the plane with normal vector vecN
            c= h*vecN
            #x-axis - c, the projection
            px= op-c
            #given the normal vector vecN and the projected x-axis px, the
            # y- axis is cross(vecN,px)
            py= np.cross(vecN,px)
            #projection of data onto space span{px,py}
            pcx=cloud_pca@px.T
            pcy=cloud_pca@py.T
            #pdata = np.array([pcx,pcy])
            #represent data in polar coordinates
            rho,theta=cart2pol(pcx,pcy)

            #main function, count points in bins
            hist= np.histogram2d(theta.T,rho.T,[thetaList,rhoList])[0]
            hist=hist/cloud_pca.shape[0]
            A.append(hist.T.ravel())
    return np.array(A)

def M2DP(cloud, numT, numR, numP, numQ):
    #key parameter
    # # number of bins in theta, the 't' in paper
    # numT = 16
    # #number of bins in rho, the 'l' in paper
    # numR = 8
    # # number of azimuth angles, the 'p' in paper
    # numP = 2
    # # number of elevation angles, the 'q' in paper
    # numQ = 4

    '''3D rotate input data so that
    x-axis and y-axis are the
    1st and 2nd PCs of data respectively
    rotation invariant'''
    pca=PCA()
    cloud_pca=pca.fit_transform(cloud)
    np.negative(cloud_pca[:,1])
    np.negative(cloud_pca[:,2])
    half_pi=math.pi/2
    azimuthList =np.linspace(-half_pi,half_pi,numP)
    elevationList = np.linspace(0,half_pi,numQ)
    #get the farthest point distance
    rho2=np.sum(np.square(cloud_pca),axis=1)
    maxRho=np.sqrt(rho2.max())
    #main function, get the signature matrix A
    A=GetSignatureMatrix(azimuthList, elevationList, cloud_pca, numT, numR, maxRho)
    #run SVD on A and use [u0,v0] as the final output
    u,s,v = np.linalg.svd(A);
    desM2DP = np.append(u[:,0],v.T[:,0])
    return desM2DP,A


import numpy as np
from scipy.stats import describe

def custom_descriptor(point_cloud, height, num_bins=5):
    # Extract the height values from the point cloud
    point_cloud[:,2]-=np.min(point_cloud[:,2])
    heights = point_cloud[:, 2]

    # Determine the height range
    min_height, max_height = 0, height
    bin_width = (max_height - min_height) / num_bins

    # Create an array to store statistical features for each bin
    descriptors = []

    for bin_index in range(num_bins):
        # Define the bin height range
        bin_min_height = min_height + bin_index * bin_width
        bin_max_height = min_height + (bin_index + 1) * bin_width

        # Select points within the current bin
        bin_points = point_cloud[(heights >= bin_min_height) & (heights < bin_max_height)]

        if len(bin_points) > 0:
            # Compute statistical features for the bin
            # mean = np.mean(bin_points, axis=0)
            count = len(bin_points)/len(point_cloud)
            std_dev = np.std(bin_points, axis=0)
            std_dev = np.linalg.norm(std_dev)
            # print(describe(bin_points)[-2:])
            skewness, kurtosis = describe(bin_points)[-2:]
            # print(skewness.shape)
            # Concatenate statistical features into a single descriptor
            # bin_descriptor = np.concatenate(([count], std_dev, skewness, kurtosis))
            # print(skewness, kurtosis)
            # bin_descriptor = np.concatenate(([count], std_dev, skewness, [kurtosis[0]]))
            bin_descriptor = np.concatenate(([count], [std_dev]))
            # print(bin_descriptor.shape)
            descriptors.append(bin_descriptor)
        else:
            # descriptors.append(np.zeros(10))
            descriptors.append(np.zeros(2))

    return np.array(descriptors).flatten()

def custom_descriptor_color(point_cloud, color, height, num_bins=5):
    # Extract the height values from the point cloud
    point_cloud[:,2]-=np.min(point_cloud[:,2])
    heights = point_cloud[:, 2]

    # Determine the height range
    min_height, max_height = 0, height
    bin_width = (max_height - min_height) / num_bins

    # Create an array to store statistical features for each bin
    descriptors = []

    for bin_index in range(num_bins):
        # Define the bin height range
        bin_min_height = min_height + bin_index * bin_width
        bin_max_height = min_height + (bin_index + 1) * bin_width

        # Select points within the current bin
        bin_points = point_cloud[(heights >= bin_min_height) & (heights < bin_max_height)]
        bin_colors = color[(heights >= bin_min_height) & (heights < bin_max_height)]

        if len(bin_points) > 0:
            # Compute statistical features for the bin
            # mean = np.mean(bin_points, axis=0)
            count = len(bin_points)/len(point_cloud)
            std_dev = np.std(bin_points, axis=0)
            std_dev = np.linalg.norm(std_dev)
            # print(describe(bin_points)[-2:])
            skewness, kurtosis = describe(bin_points)[-2:]
            mean_color = np.mean(bin_colors,axis=0)
            # print(skewness.shape)
            # Concatenate statistical features into a single descriptor
            # bin_descriptor = np.concatenate(([count], std_dev, skewness, kurtosis))
            # print(skewness, kurtosis)
            # bin_descriptor = np.concatenate(([count], std_dev, skewness, [kurtosis[0]]))
            bin_descriptor = np.concatenate(([count], [std_dev], mean_color))
            # print(bin_descriptor.shape)
            descriptors.append(bin_descriptor)
        else:
            # descriptors.append(np.zeros(10))
            descriptors.append(np.zeros(5))

    return np.array(descriptors).flatten()
