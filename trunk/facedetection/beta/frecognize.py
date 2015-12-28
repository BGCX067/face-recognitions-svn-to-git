# -*- coding: utf-8 -*-
'''
Modul um die Methoden für das FisherFace Algorithmus umzusetzten
'''
import logging as log
import numpy as np
import numpy.linalg as la
import cv2
#np.set_printoptions(threshold=np.nan)



class FaceRecognizer(object):
    '''
    Klasse der den FisherFace Algorithmus mit gegebenen TrainingSets trainiert, eine id-Klasse voraussagt und Gesicht Images vergleicht
    '''


    def __init__(self, num_comp = 0):
        '''
        Konstruktor, der leere Variablen instanziert
        '''
        self.possible_ids = {}
        
        self.num_comp = 0
        self.W = []
        self.eig_vec_pca = []
        self.mean = None
        self.threshold = np.finfo('float').max
        self.projections = []
    def trainFisherFaces(self, face_ids, face_images):
        """Trainiert den FisherFace Algorithmus mit eine Array gegebene id-Klassen 
        und ein Liste von Gesichtimage Matrizen
        Erstellt eine Liste von Projektionen"""
        self.face_ids = face_ids
        self.projections = []
        # Liste von Bildern Matrizen werden umgeformt zu ein Matrix mit ein Image pro Zeile
        face_images = self.face_images_as_rows(face_images)
        #Berechne Eigenvektoren eig_vec_pca und Mittelwert mean mit PCA, Nummer von Komponenten muss num_comp = n-c sein
        [self.eig_vec_pca, self.mean] = self.pca(face_images, (face_images.shape[0]-len(np.unique(self.face_ids))))
        #Berechne Eigenvektoren eig_vec_lda mit LDA
        self.eig_vec_lda = self.lda(self.project(face_images,self.eig_vec_pca, self.mean), self.face_ids, self.num_comp)
        
        #Tranformations Matrix W, der ein Image Sample in ein (c-1) Dimensionen Space projeziert wird berechnet
        self.W = np.asmatrix(np.dot(self.eig_vec_pca, self.eig_vec_lda))
        #Jedes Bild wird projeziert und die Projektion wird zu eine Liste Projektionen hinzugefügt
        for fi in face_images:
            self.projections.append(self.project(fi.reshape(1,-1),self.W,self.mean))
        #Eigenvektoren und Rekonstruktion ausführen
        #self.form(face_images)
            
    def predict(self,unknown_face):
        """Bekommt das zu testende vorbearbeitete Gesichtsbild und gleicht es mit der Datenbank ab.
        return -> face_id
        Die ID der Person der dieses Gesichtsbild am aehnlichsten ist.
        
        """
        max_dist = 1.1
        face_id = -1
        min_dist = max_dist

        #Unbekannter Gesicht wird in unser Projectionsmatrix projeziert
        unknown_face = self.project(unknown_face.reshape(1,-1), self.W, self.mean)        
        #der am nähesten Nachbar, wird berechnet, 
        #in dem die kurzeste Distanz aus die Projektionen,basiert auf den euklidischen Distanz, berechnet wird
        for p in range(len(self.projections)):
            d = self.cosine_distance(self.projections[p], unknown_face)
            if d < min_dist:
                min_dist = d
                face_id = self.face_ids[p]
                log.debug('min_dist: %s id: %s' % (min_dist, face_id))
        return face_id
    
    def project(self, face, W, u = None):
        """Berechnet ein Projection in dem p = W(face-u) """
        if u is None:
            return np.dot(face,W)
        return np.dot(face-u,W)
    
    def face_images_as_rows(self,face_images):
        """Formt jede Image Matrix in der Gesichtimage Matrizen Liste zu eine Zeile einer Matrix"""
        if len(face_images) == 0:
            return np.asarray([])
        data_matrix = np.empty((0,face_images[0].size), dtype=face_images[0].dtype)
        for fi in face_images:
            data_matrix = np.append(data_matrix, fi.reshape(1,-1), axis=0)
        return np.asmatrix(data_matrix)

    def pca(self, face_images,num_comp = 0):
        """Setzt den Principal Component Analysis Algorithmus mit ein Gesicht Image Matrix um """
        [n,d] = face_images.shape
        if num_comp<=0 or num_comp > n:
            num_comp = n
        #Errechne der Mittelwert von alle Klassen aus alle Bilder(der Gesicht Image Matrix)
        mean = face_images.mean(axis=0)
        face_images = face_images - mean
        [eigenvectors, eigenvalues, var] = la.svd(face_images.T, full_matrices = False)
        #Sortiere, von eigenwerte abhängig, die eigenwerte und eigenvektoren absteigend 
        sort_eigen = np.argsort(-eigenvalues)
        eigenvectors = eigenvectors[:,sort_eigen]
        #eigenvalues = eigenvalues[sort_eigen].copy()
        
        #Schneide Eigenvektoren ab Anzahl von Komponenten ab, wollen nur non-null comp haben
        eigenvectors = eigenvectors[:,:num_comp].copy()
        #eigenvalues = eigenvalues[:num_comp].copy()
        return eigenvectors, mean

    def lda(self, face_images, face_ids, num_comp = 0):
        """Setzt das Lineare Discriminant Analysis mit ein Gesicht Image Matrix und ein Array der id-Klassen um """
        #Die verschieden Klassen an sich
        c = np.unique(face_ids)
        [n,d] = face_images.shape
        if num_comp<=0 or num_comp>(len(c)-1):
            num_comp = len(c)-1

        #Errechne der Mittelwert von alle Klassen aus alle Bilder(der Gesicht Image Matrix)
        total_mean = face_images.mean(axis=0)
        sb = np.zeros((d,d), dtype = np.float32)
        sw = np.zeros((d,d), dtype = np.float32)
        
        for i in c:
            #Alle Gesicht Images einer Klasse
            face_i = face_images[np.where(face_ids==i)[0],:]
            #Mittelwert einer Klasse
            mean_i = face_i.mean(axis=0)
            #Zwischen-Klassen Verteilung
            sb = sb + n * np.dot((mean_i-total_mean).T, (mean_i-total_mean))
            #Innerhalb-Klassen Verteilung
            sw = sw +np.dot((face_i-mean_i).T,(face_i-mean_i))
        e = np.dot(la.inv(sw),sb)
        [eigenvalues, eigenvectors] = la.eig(e)
        #Sortiere von eigenwerte abhängig die eigenwerte und eigenvektoren absteigend 
        sort_eigen = np.argsort(-eigenvalues.real)
        eigenvectors = eigenvectors[:,sort_eigen]
        #eigenvalues = eigenvalues[sort_eigen]
        #Schneide Eigenvektoren ab Anzahl von Komponenten ab, wollen nur non-null comp haben
        eigenvectors = np.array(eigenvectors[:,:num_comp].real, dtype=np.float32, copy = True)
        #eigenvalues = np.array(eigenvalues[:num_comp].real, dtype=np.float32, copy = True)
        return eigenvectors

    def euclidean_distance(self,p,q):
        """ Errechnet den euklidischen Distanz zweier Projektion Matrizen """
        p = np.asarray(p).flatten()
        q = np.asarray(q).flatten()
        return np.sqrt(np.sum(np.power((p-q),2)))

    def cosine_distance(self, p, q):
        """ Errechnet den Kosinus Distanz zweier Projektion Matrizen """
        p = np.asarray(p).flatten()
        q = np.asarray(q).flatten()
        
        return 1-(np.dot(p,q)/(la.norm(p)*la.norm(q)))

    def get_similar(self,unknown_face):
        """Errechnet die Ähnlichkeit einer rekonstruierter Gesicht mit den unbekannter Gesicht mit den L2Error """
        average_face = self.reconstruct_face(unknown_face)
        l2 = cv2.norm(unknown_face,average_face,cv2.NORM_L2)
        similar = l2/(unknown_face.shape[0]*unknown_face.shape[1])
        return similar
    
    def reconstruct_face(self,face):
        """ Rekonstruiert ein Gesicht mit den Eigenvektoren von den PCA Algorithmus """
        p = self.project(face.reshape(1,-1),self.eig_vec_pca, self.mean)
        r = self.reconstruct(p, self.eig_vec_pca, self.mean)
        r = r.reshape(face.shape)
        return np.asmatrix(r, dtype = np.uint8)
    
    def reconstruct(self,projection,W,u = None):
        """Berechnet ein Rekonstruktion mit face = W(p+u) """
        if u is None:
            return np.dot(projection, W.T)
        return np.dot(projection, W.T)+u
    
    def form_gray_face(self,average_face):
        """Unbenutzte Methode, soll average_face zu Graustufenbilder umformen"""
        correct_form = average_face[:].reshape(70,70)
        minP = float(np.min(correct_form))
        maxP = float(np.max(correct_form))
        correct_form = correct_form-minP
        correct_form = correct_form/(maxP-minP)
        correct_form = np.asmatrix(correct_form, dtype=np.uint8)
        return correct_form
    def form(self,face_images):
        """Unbenutzte Method, verbildlicht die ersten 16 Eigenvektoren von PCA und Rekonstruktion von PCA
        und die Eigenvectoren von Fisherface und Rekonstruktion von Fisherface"""
        count = 1
        #Fisherfaces Eigenvektors
        for i in xrange(min(self.W.shape[1], 16)):
            w_row = self.W[:,i].reshape(face_images[0].shape)
            ff = self.form_gray_face(w_row)
            cv2.imwrite("fisherface"+str(count)+".png",ff)
            count +=1
        #Fisherfaces Rekonstruktion
        for i in xrange(min(self.W.shape[1], 16)):
            if i != 10:
                w_row = self.W[:,i].reshape(-1,1)
                projection = self.project(face_images[10].reshape(1,-1), w_row,self.mean)
                reconstruct = self.reconstruct(projection,w_row, self.mean)
                reconstruct = reconstruct.reshape(face_images[0].shape)
                ff = self.form_gray_face(reconstruct)
                cv2.imwrite("fisherface"+str(count)+".png",ff)
                count +=1
        #Eigenfaces Eigenvektors
        for i in xrange(min(len(face_images), 16)):
            w_row = self.eig_vec_pca[:,i].reshape(face_images[0].shape)
            ff = self.form_gray_face(w_row)
            cv2.imwrite("eigenface"+str(count)+".png",ff)
            count +=1
        #Eigenfaces Rekonstruktion
        steps=[i for i in xrange(0, min(len(face_images), 320), 20)]
        for i in xrange(min(len(steps), 16)):
                n = steps[i]
                projection = self.project(face_images[10].reshape(1,-1), self.eig_vec_pca[:,0:n],self.mean)
                reconstruct = self.reconstruct(projection,self.eig_vec_pca[:,0:n], self.mean)
                reconstruct = reconstruct.reshape(face_images[0].shape)
                ff = self.form_gray_face(reconstruct)
                cv2.imwrite("eigenface"+str(count)+".png",ff)
                count +=1
