import numpy as np
from copy import deepcopy

def center_of_range(two_d_array):
    return np.max(two_d_array, axis=0) - np.min(two_d_array, axis=0)

def get_one_hot(vector_of_inds):
    n_values = np.max(vector_of_inds) + 1
    return np.eye(n_values)[vector_of_inds]

class MultiChannelKMeans():
    def __init__(self, image_vector, k, convergence=0.001, max_iter=20, max_init=100):
        assert(type(image_vector) == np.ndarray)
        assert(len(image_vector.shape) == 2)
        assert(type(k) == type([]))
        assert(len(k) == image_vector.shape[1])
        assert(type(max_iter) == type(0))
        assert(type(convergence) == type(0.0))
        assert(type(max_init)) == type(0)

        self.image_vector = image_vector
        self.k = k
        self.n_fit_colors = k
        self.max_iter = max_iter
        self.convergence = convergence
        self.max_init = max_init

        self.out_image_vector = np.zeros(self.image_vector.shape)

        self.n_pixels = self.image_vector.shape[0]
        self.n_color_channels = self.image_vector.shape[1]

        # self.cluster_centers_ = np.zeros((self.k, self.n_color_channels))
        # self.labels_ = np.ones(self.n_pixels).astype(np.int32)

        self.channel_inds = []
        self.image_vectors = []
        self.kms = []
        for i in range(self.n_color_channels):
            self.channel_inds.append(np.where(np.argmax(image_vector, axis=1)==i))
            self.image_vectors.append(self.image_vector[self.channel_inds[i]])
            self.kms.append(ImageKMeans(self.image_vectors[i], self.k[i], convergence, max_iter, max_init))

    def set_custom_colors(self, custom_colors):
        assert(type(custom_colors) == np.ndarray)
        self.custom_colors = []
        for i in range(self.n_color_channels):
            inds = np.where(np.argmax(custom_colors, axis=1)==i)
            self.kms[i].set_custom_colors(custom_colors[inds])

    def extract(self):
        for i in range(self.n_color_channels):
            self.out_image_vector[self.channel_inds[i]] = self.kms[i].out_image_vector

    def fit(self):
        for i in range(self.n_color_channels):
            self.kms[i].fit()
        self.extract()

    def get_image(self, dim1, dim2):
        return self.out_image_vector.reshape(dim1, dim2, self.n_color_channels)


class ImageKMeans():
    def __init__(self, image_vector, k, convergence=0.001, max_iter=20, max_init=100):
        assert(type(image_vector) == np.ndarray)
        assert(len(image_vector.shape) == 2)
        assert(type(k) == type(0))
        assert(type(max_iter) == type(0))
        assert(type(convergence) == type(0.0))
        assert(type(max_init)) == type(0)

        self.image_vector = image_vector
        self.k = k
        self.n_fit_colors = k
        self.max_iter = max_iter
        self.convergence = convergence
        self.max_init = max_init

        self.n_pixels = self.image_vector.shape[0]
        self.n_color_channels = self.image_vector.shape[1]

        self.cluster_centers_ = np.zeros((self.k, self.n_color_channels))
        self.labels_ = np.ones(self.n_pixels).astype(np.int32)

    def centroids_ok(self):
        ### check to see if there are any duplicate colors
        for i in range(self.k):
            if self.cluster_centers_[i] in self.cluster_centers_[:i] or self.cluster_centers_[i] in self.cluster_centers_[i+1:]:
                return False

        ### check that each centroid has at least one pixel assigned to it
        self.assign_points_to_centroids()
        for color_idx in range(self.k):
            if color_idx not in self.labels_:
                if color_idx < self.n_fit_colors:
                    return False
                else:
                    raise Exception(f'Custom color index {color_idx - self.n_fit_colors} has no points assigned to it')
        return True

    def init_centroids(self):
        for j in range(self.max_init):
            for i in range(self.n_fit_colors):
                self.cluster_centers_[i,:] = self.image_vector[np.random.randint(self.n_pixels)]
                if self.centroids_ok():
                    return

    def set_custom_colors(self, custom_colors):
        assert(type(custom_colors) == np.ndarray)
        assert(custom_colors.shape[1] == self.n_color_channels)
        assert(custom_colors.shape[0] <= self.k)
        self.custom_colors = custom_colors
        self.n_custom_colors = self.custom_colors.shape[0]
        self.n_fit_colors = self.k - self.n_custom_colors
        self.cluster_centers_[self.n_fit_colors:,:] = self.custom_colors
        return

    def assign_points_to_centroids(self):
        '''
        image_vector is shape (n_pixels, n_color_channels). n_color_channels is usually 3 (r,g,b)
        repeated_pixels is image_vector reshaped to (n_pixels, k, n_color_channels)
        cluster_centers_ is shape (k, n_color_channels)
        repeated_centroids is cluster_centers_ reshaped to (n_pixels, k, n_color_channels)
        subtract these
        then take the norm along the axis that corresponds to the rgb values (n_color_channels)
        then take the argmin along the axis that corresponds to the cluster centers (k)
        '''
        repeated_pixels = np.expand_dims(self.image_vector, 1)
        repeated_pixels = repeated_pixels.repeat(self.k, axis=1)
        repeated_centroids = np.expand_dims(self.cluster_centers_, 0)
        repeated_centroids = repeated_centroids.repeat(self.n_pixels, 0)
        differences = repeated_pixels - repeated_centroids
        distances = np.linalg.norm(differences, axis=2)
        self.labels_ = np.argmin(distances, axis=1)
        # for i in range(self.n_pixels):
        #     distances = np.linalg.norm(self.cluster_centers_ - self.image_vector[i,:])
        #     self.labels_[i] = np.argmin(distances)

    def move_centroids_to_center(self):
        old_centroids = deepcopy(self.cluster_centers_)
        print('Before:')
        print(self.cluster_centers_)
        for i in range(self.n_fit_colors):
            current_indices = np.where(self.labels_ == i)
            pixel_values = self.image_vector[current_indices]
            print(pixel_values.shape[0])
            center = np.mean(pixel_values, axis=0)
            # center = center_of_range(pixel_values)
            self.cluster_centers_[i,:] = center
        distances = np.linalg.norm(self.cluster_centers_ - old_centroids, axis=1)
        print('After:')
        print(self.cluster_centers_, '\n')
        return np.max(distances)

    def fit(self):
        self.init_centroids()
        for i in range(self.max_iter):
            print(f'\n{i}')
            self.assign_points_to_centroids()
            max_centroid_movement = self.move_centroids_to_center()
            print(max_centroid_movement)
            if max_centroid_movement <= self.convergence:
                break
        self.extract()

    def extract(self):
        one_hots = get_one_hot(self.labels_)
        out_image_vector = np.dot(one_hots, self.cluster_centers_)
        self.out_image_vector = out_image_vector

    def get_image(self, dim1, dim2):
        return self.out_image_vector.reshape(dim1, dim2, self.n_color_channels)
