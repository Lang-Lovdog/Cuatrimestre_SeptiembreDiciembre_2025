# LBP_feat_h.py
import warnings

import os
import numpy           as     np                    #type: ignore
import pandas          as     pd                    #type: ignore
import cv2             as     cv                    #type: ignore
from   skimage.feature import local_binary_pattern  #type: ignore
from   skimage.feature import graycoprops           #type: ignore
from   skimage.feature import graycomatrix          #type: ignore


warnings.filterwarnings('ignore')  # Suppress all warnings


class LBPFeatures:
    features = [
        "File",
        "mean",
        "variance",
        "correlation",
        "contrast",
        "homogeneity",
        "class"
    ]
    verbose = 1


    def __init__(self, data_loader=None):
        """
        Initialize LBP feature extractor.
        Args:
            data_loader (LovdogDataFrames): A loaded data manager object.
        """
        self.data_loader = data_loader
        self.LBP = {}
        self.features_table = pd.DataFrame(columns=self.features)
        
        # LBP parameters with defaults
        self.ratio = 1
        self.points = 8
        self.method = "uniform"
        self.window_size = [3]
        
        # Store images if needed (for file-based processing)
        self.images = {}

    def set_data_loader(self, data_loader):
        """Set the data loader after initialization."""
        self.data_loader = data_loader

    def set_parameters(self, ratio=None, points=None, method=None, window_size=None):
        """Configure LBP parameters."""
        if ratio is not None:
            self.ratio = ratio
        if points is not None:
            self.points = points
        if method is not None:
            self.method = method
        if window_size is not None:
            if not isinstance(window_size, list):
                window_size = [window_size]
            self.window_size = window_size

    def compute_lbp_from_data_loader(self):
        """
        Compute LBP features using data from the LovdogDataFrames object.
        This is the NEW main method that uses your unified data structure.
        """
        if self.data_loader is None:
            raise ValueError("No data loader provided. Call set_data_loader() first.")
        
        frames_dict = self.data_loader.get_data_dict()
        class_names = self.data_loader.get_class_names()

        bin_total = 25
        bin_value = 0
        
        print("[INFO] Computing LBP features from loaded data...")
        
        for class_name in class_names:
            print(f"  Processing class: {class_name}")
            self.LBP[class_name] = []
            fd_size   = len(frames_dict[class_name])
            bin_value = fd_size // bin_total
            fd_count  = 0
            
            for i, frame in enumerate(frames_dict[class_name]):
                # Convert frame to uint8 if needed (assuming your frames are 2D arrays)
                if isinstance(frame, np.ndarray):
                    # Normalize and convert to uint8 for LBP processing
                    frame_normalized = cv.normalize(frame, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
                    lbp = local_binary_pattern(frame_normalized, self.points, self.ratio, self.method).astype(np.uint8)
                    
                    # Use a generic filename or index since we're not loading from files directly
                    file_id = f"{class_name}_{i:04d}"
                    self.LBP[class_name].append({"File": file_id, "LBP": lbp, "class": class_name})
                
                # Optional: Keep the old file-based approach as a fallback
                elif isinstance(frame, str) and os.path.exists(frame):
                    print(f"    Loading from file: {frame}")
                    img = cv.imread(frame, cv.IMREAD_GRAYSCALE)
                    if img is not None:
                        lbp = local_binary_pattern(img, self.points, self.ratio, self.method).astype(np.uint8)
                        self.LBP[class_name].append({"File": os.path.basename(frame), "LBP": lbp, "class": class_name})
                fd_count += 1
                if fd_count % bin_value == 0:
                    print(".", end="", flush=True)

            print("\n")


    # Keep your original file-based methods for compatibility
    def get_images(self):
        """Original file-based image loading (for backward compatibility)"""
        if len(self.images) > 0:
            self.images = {}

        for sample in self.samples:
            self.images[sample["class"]] = [
                os.path.join(sample["path"], file)
                for file in os.listdir(sample["path"])
                if file.endswith(self.file_pattern)
            ]

    def compute_lbp_from_files(self):
        """Original file-based LBP computation (for backward compatibility)"""
        for key in self.images:
            self.LBP[key] = []
            for file in self.images[key]:
                img = cv.imread(file, cv.IMREAD_GRAYSCALE)
                lbp = local_binary_pattern(img, self.points, self.ratio, self.method).astype(np.uint8)
                self.LBP[key].append({"File": os.path.basename(file), "LBP": lbp, "class": key})

    # Keep your static methods (they're good!)
    @staticmethod
    def append_features_row(row, dataframe):
        if not isinstance(row, pd.DataFrame):
            row = pd.DataFrame(row)
        return pd.concat([dataframe, row], ignore_index=True, axis=0)

    @staticmethod
    def get_features_from_window(window):
        features = {}
        glcm = graycomatrix(window, [1], [0], symmetric=True, normed=True, levels=59)
        features["mean"] = graycoprops(glcm, 'mean')[0][0]
        features["variance"] = graycoprops(glcm, 'variance')[0][0]
        features["correlation"] = graycoprops(glcm, 'correlation')[0][0]
        features["contrast"] = graycoprops(glcm, 'contrast')[0][0]
        features["homogeneity"] = graycoprops(glcm, 'homogeneity')[0][0]
        return features

    @staticmethod
    def lbp_window_analysis(lbp_img, window_size, _class=None):
        x = 0
        y = 0
        lbp_image = lbp_img["LBP"]
        lbp_file = lbp_img["File"]
        analysis = []
        run_boggle = [ "-", "\\", "|", "/" ]
        idx_boggle = 0

        if window_size < 0:
            lbp_feat = LBPFeatures.get_features_from_window(lbp_image)
            lbp_feat["File"] = lbp_file
            if _class is not None:
                lbp_feat["class"] = _class
            analysis.append(lbp_feat)
            return pd.DataFrame(analysis, columns=LBPFeatures.features)
        
        for x in range(lbp_image.shape[0] - window_size):
            for y in range(lbp_image.shape[1] - window_size):
                window = lbp_image[x:x + window_size, y:y + window_size]
                lbp_feat = LBPFeatures.get_features_from_window(window)
                lbp_feat["File"] = lbp_file
                if _class is not None:
                    lbp_feat["class"] = _class
                analysis.append(lbp_feat)
                print(run_boggle[idx_boggle], end="", flush=True)
                idx_boggle = idx_boggle + 1 if idx_boggle < 3 else 0
                print("\b", end="", flush=True)
        
        return pd.DataFrame(analysis, columns=LBPFeatures.features)


    def compute_features_from_lbp(self):
        """Compute features from LBP images (works with both old and new data sources)"""
        bin_total = 25
        bin_value = 0
        for c in self.LBP.keys():
            print(f"Processing {c} images...")
            fd_size = len(self.LBP[c])
            bin_value = fd_size // bin_total
            fd_count = 0
            for i in self.LBP[c]:
                for ws in self.window_size:
                    result = self.lbp_window_analysis(i, ws, _class=c)
                    self.features_table = LBPFeatures.append_features_row(result, self.features_table)
                if fd_count % bin_value == 0:
                    print("*", end="", flush=True)
                fd_count += 1
            print("\n", end="", flush=True)
        print(self.features_table)


    def save_to_csv(self, csv_path=None):
        """Save features to CSV file."""
        if csv_path is None and hasattr(self, 'csv_name'):
            csv_path = self.csv_name
        elif csv_path is None:
            csv_path = "LBP_Features.csv"
            
        self.features_table.to_csv(csv_path, index=False)
        print(f"Features saved to: {csv_path}")

    def get_features_dataframe(self):
        """
        Return the computed features as a pandas DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame containing all extracted features with columns:
                         ['File', 'mean', 'variance', 'correlation', 'contrast', 
                          'homogeneity', 'class']
        """
        return self.features_table.copy()  # Return a copy to prevent accidental modification

    def get_ml_data(self, include_file=False):
        """
        Return features and labels in standard scikit-learn format.
        
        Args:
            include_file (bool): Whether to include the 'File' column in features
        
        Returns:
            tuple: (X_features, y_labels) where:
                   X_features: numpy array or DataFrame of features
                   y_labels: numpy array or Series of class labels
        """
        if include_file:
            X = self.features_table.drop('class', axis=1)
        else:
            X = self.features_table.drop(['File', 'class'], axis=1)
        
        y = self.features_table['class']
        return X, y
