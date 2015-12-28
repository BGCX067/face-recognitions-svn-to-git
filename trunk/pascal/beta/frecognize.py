# -*- coding: utf-8 -*-
'''
Created on 27.01.2014

@author: jjoch001
'''
import database as m
import numpy as np
import numpy.linalg as la
import cv2
#np.set_printoptions(threshold=np.nan)

import logging as log

class FaceRecognizer(object):
    '''
    classdocs
    '''


    def __init__(self, num_comp = 0):
        '''
        Constructor
        '''
        self.possible_ids = {}
        
        self.num_comp = 0
        self.W = []
        self.eig_vec_pca = []
        self.mu = None
        self.threshold = np.finfo('float').max
        self.projections = []
    def trainFisherFaces(self, face_ids, face_images):
        self.face_ids = face_ids
        self.projections = []
        # Matrix von Bildern werden in eine Zeile gespeichert
        face_images = self.face_images_as_rows(face_images)
        #Berechne Eigenvektoren eig_vec_pca und Mittelwert mean mit PCA, Nummer von Komponenten muss num_comp = n-c sein
        [self.eig_vec_pca, self.mu] = self.pca(face_images, (face_images.shape[0]-len(np.unique(self.face_ids))))
        #Berechne Eigenvektoren eig_vec_lda mit LDA
        eig_vec_lda = self.lda(self.project(face_images,self.eig_vec_pca, self.mu), self.face_ids, self.num_comp)
        
        #Tranformations Matrix W, der ein Image Sample in ein (c-1) Dimensionen Space projeziert wird berechnet
        self.W = np.asmatrix(np.dot(self.eig_vec_pca, eig_vec_lda))
        #Jedes Bild wird projeziert und die Projektion wird zu eine Liste Projectionen hinzugefügt
        for fi in face_images:
            self.projections.append(self.project(fi.reshape(1,-1),self.W,self.mu))
            
    #der am nähesten Nachbar, wird berechnet, 
    #in dem die kurzeste Distanz aus die Projektionen,basiert auf den euklidischen Distanz, berechnet wird
    def predict(self,unknown_face):
        """Bekommt das zu testende vorbearbeitete Gesichtsbild und gleicht es mit der Datenbank ab.
        return -> face_id
        Die ID der Person der dieses Gesichtsbild am aehnlichsten ist.
        
        """
        #TODO: Initial Wert von min_dist verfeinern
        max_dist = 1.1 #np.finfo('float').max
        face_id = -1
        min_dist = max_dist
        
        #Unbekannter Gesicht wird in unser Projectionsmatrix projeziert
        unknown_face = self.project(unknown_face.reshape(1,-1), self.W, self.mu)
        
        
        for p in range(len(self.projections)):
            
            d = self.cosine_distance(self.projections[p], unknown_face)
            if d < min_dist:
                min_dist = d
                face_id = self.face_ids[p]
                #percent = 1.0 - (min_dist/max_dist)
#                 log.debug('min_dist: %s id: %s' % (min_dist, face_id))
        return face_id
    
    #y = W^T(X-u) schauen ob W.T korrekt ist. 
    def project(self, face, W, u = None):
        if u is None:
            return np.dot(face,W)
        return np.dot(face-u,W)
    
    #Jede Reihe is ein flache image matrix
    def face_images_as_rows(self,face_images):
        if len(face_images) == 0:
            return np.asarray([])
        data_matrix = np.empty((0,face_images[0].size), dtype=face_images[0].dtype)
        for fi in face_images:
            data_matrix = np.vstack((data_matrix, np.asarray(fi).reshape(1,-1)))
        return np.asmatrix(data_matrix)
    #Principal Component Analysis
    def pca(self, face_images,num_comp = 0):
        [n,d] = face_images.shape
        if num_comp<=0 or num_comp > n:
            num_comp = n
        #Errechne der Mittelwert von alle Klassen und ihre Samples(Images)
        mean = face_images.mean(axis=0)
        face_images = face_images - mean
        #[eigenvectors, eigenvalues, var] = la.svd(face_images.T, full_matrices = False)
        if n>d:
            C = np.dot(face_images.T,face_images)
            [eigenvalues,eigenvectors] = np.linalg.eigh(C)
        else:
            C = np.dot(face_images,face_images.T)
            [eigenvalues,eigenvectors] = np.linalg.eigh(C)
            eigenvectors = np.dot(face_images.T,eigenvectors)
            for i in xrange(n):
                eigenvectors[:,i] = eigenvectors[:,i]/np.linalg.norm(eigenvectors[:,i])
        #Sortiere von eigenwerte abhängig die eigenwerte und eigenvektoren absteigend 
        sort_eigen = np.argsort(-eigenvalues)
        eigenvectors = eigenvectors[:,sort_eigen]
        #eigenvalues = eigenvalues[sort_eigen].copy()
        #Schneide Eigenvektoren ab Anzahl von Komponenten ab, wollen nur non-null comp haben
        #eigenvalues = eigenvalues[:num_comp].copy()
        eigenvectors = eigenvectors[:,:num_comp].copy()
        return eigenvectors, mean
    #Lineare Discriminant Analysis
    def lda(self, face_images, face_ids, num_comp = 0):
        #Die verschieden Klassen an sich
        c = np.unique(face_ids)
        [n,d] = face_images.shape
        if num_comp<=0 or num_comp>(len(c)-1):
            num_comp = len(c)-1
        #Mittelwert aus alle Klassen
        total_mean = face_images.mean(axis=0)
        sb = np.zeros((d,d), dtype = np.float32)
        sw = np.zeros((d,d), dtype = np.float32)
        
        for i in c:
            #Images (Samples) einer Klasse
            face_i = face_images[np.where(face_ids==i)[0],:]
            #Mittelwert einer Klasse
            mean_i = face_i.mean(axis=0)
            #Zwischen-Klassen Verteilung
            sb = sb + n * np.dot((mean_i-total_mean).T, (mean_i-total_mean))
            #Innerhalb-Klassen Verteilung
            sw = sw +np.dot((face_i-mean_i).T,(face_i-mean_i))
        #np.svd kann nicht verwendet werden weil sw und sw nicht umbedingt symmetrische matrixen ergeben 
        [eigenvalues, eigenvectors] = la.eig(la.inv(sw)*sb)
        #Sortiere von eigenwerte abhängig die eigenwerte und eigenvektoren absteigend 
        sort_eigen = np.argsort(-eigenvalues.real)
        
        eigenvectors = eigenvectors[:,sort_eigen]
        #eigenvalues= eigenvalues[sort_eigen]
        #Schneide Eigenvektoren ab Anzahl von Komponenten ab, wollen nur non-null comp haben
        #eigenvalues = np.array(eigenvalues[:num_comp].real, dtype=np.float32, copy = True)
        eigenvectors = np.array(eigenvectors[:,:num_comp].real, dtype=np.float32, copy = True)
        return eigenvectors
    #Euklidischen Distanz zweier Projektion Matrizen
    def euclidean_distance(self,p,q):
        p = np.asarray(p).flatten()
        q = np.asarray(q).flatten()
        return np.sqrt(np.sum(np.power((p-q),2)))

    def cosine_distance(self, p, q):
        p = np.asarray(p).flatten()
        q = np.asarray(q).flatten()
        return 1-(np.dot(p,q)/(la.norm(p)*la.norm(q)))
        #return -np.dot(p.T,q) / (np.sqrt(np.dot(p,p.T)*np.dot(q,q.T)))
    def get_confidence(self,unknown_face):
        average_face = self.reconstruct_face(unknown_face)
        l2 = cv2.norm(unknown_face,average_face,cv2.NORM_L2)
        similar = l2/(unknown_face.shape[0]*unknown_face.shape[1])
        return similar
    def reconstruct_face(self,face):
        p = self.project(face.reshape(1,-1),self.eig_vec_pca, self.mu)
        r = self.reconstruct(p, self.eig_vec_pca, self.mu)
        r = r.reshape(face.shape)
        return np.asmatrix(r, dtype = np.uint8)
    
    def reconstruct(self,projection,W,u = None):
        if u is None:
            return np.dot(projection, W.T)
        return np.dot(projection, W.T)+u
    def form_average_face(self,average_face,face_size):
        correct_form = average_face[:].reshape(face_size)
        minP = float(np.min(correct_form))
        maxP = float(np.max(correct_form))
        correct_form = correct_form-minP
        correct_form = correct_form/(maxP-minP)
        correct_form =255* correct_form
        #correct_form =255* ((correct_form-minP)/(maxP-minP))
        correct_form = np.asmatrix(correct_form, dtype=np.uint8)
        print correct_form
        cv2.namedWindow('averageface')
        cv2.imshow('averageface', correct_form)
        return correct_form