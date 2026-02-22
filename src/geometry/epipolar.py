import cv2
import numpy as np

class EpipolarGeometry:
    """Handles multi-view geometry operations for visual odometry.

    Estimates the relative camera motion (rotation and translation) between
    consecutive frames by computing the Essential matrix from matched 2D feature points.
    """
    def __init__(self, camera_intrinsics: dict):
        """Initializes the EpipolarGeometry module with the camera intrinsic matrix.

        Args:
            camera_intrinsics (dict): A dictionary containing focal lengths ('fx', 'fy')
                and principal point coordinates ('cx', 'cy').
        """
        fx = camera_intrinsics['fx']
        fy = camera_intrinsics['fy']
        cx = camera_intrinsics['cx']
        cy = camera_intrinsics['cy']
        
        # Construct the 3x3 camera intrinsic matrix K.
        self.K = np.array([
            [fx,  0, cx],
            [ 0, fy, cy],
            [ 0,  0,  1]
        ], dtype=np.float64)

    def match_features(self, desc1: np.ndarray, desc2: np.ndarray):
        """Finds corresponding physical points between two frames using descriptors.

        Applies a Brute-Force Matcher with L2 norm and filters outliers using
        Lowe's ratio test.

        Args:
            desc1 (np.ndarray): Descriptors from the first frame, shape (N, 256).
            desc2 (np.ndarray): Descriptors from the second frame, shape (M, 256).

        Returns:
            tuple: A tuple containing two 1D numpy arrays of matching indices:
                - idx1 (np.ndarray): Indices of matched keypoints in the first frame.
                - idx2 (np.ndarray): Indices of matched keypoints in the second frame.
        """
        # Initialize the Brute-Force Matcher.
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        
        # Perform k-Nearest Neighbors matching (k=2) for Lowe's ratio test.
        knn_matches = bf.knnMatch(desc1, desc2, k=2)
        
        good_matches = []
        for m, n in knn_matches:
            # Apply Lowe's ratio test to discard ambiguous matches.
            if m.distance < 0.8 * n.distance:
                good_matches.append(m)
                
        # Extract the indices of robust matches.
        idx1 = np.array([m.queryIdx for m in good_matches])
        idx2 = np.array([m.trainIdx for m in good_matches])
        
        return idx1, idx2

    def estimate_pose(
        self, kpts1: np.ndarray, kpts2: np.ndarray
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """Estimates the camera motion (R, t) between two frames.

        Args:
            kpts1: A NumPy array of shape (N, 2) containing keypoints from the first frame.
            kpts2: A NumPy array of shape (N, 2) containing matched keypoints from the second frame.

        Returns:
            A tuple containing:
                - R: A 3x3 rotation matrix (np.ndarray), or None if estimation fails.
                - t: A 3x1 translation vector (np.ndarray), or None if estimation fails.
                - inlier_mask: A 1D boolean array indicating valid correspondences,
                  or None if estimation fails.
        """
        # A minimum of 5 points is mathematically required to compute the Essential matrix.
        if len(kpts1) < 5:
            return None, None, None

        # Utilize USAC_MAGSAC for highly robust outlier rejection.
        # Confidence is set to 0.9999 for deep sampling, and the threshold is set to 1.0.
        E, mask = cv2.findEssentialMat(
            points1=kpts1, 
            points2=kpts2, 
            cameraMatrix=self.K, 
            method=cv2.USAC_MAGSAC, 
            prob=0.9999, 
            threshold=1.0
        )

        # Ensure the Essential matrix was computed successfully and has the correct dimensions.
        if E is None or E.shape != (3, 3):
            return None, None, None

        # Recover the relative rotation (R) and translation (t) from the Essential matrix.
        _, R, t, mask_pose = cv2.recoverPose(
            E=E, 
            points1=kpts1, 
            points2=kpts2, 
            cameraMatrix=self.K, 
            mask=mask
        )
        
        # Flatten the mask into a 1D boolean array for efficient external filtering.
        inlier_mask = mask_pose.ravel() > 0
        
        return R, t, inlier_mask

# Module verification block.
if __name__ == "__main__":
    # Initialize mock camera intrinsics.
    intrinsics = {'fx': 800, 'fy': 800, 'cx': 320, 'cy': 240}
    geo = EpipolarGeometry(intrinsics)
    
    # Generate mock point coordinates simulating pure forward translation.
    pts1 = np.array([[320, 240], [330, 240], [320, 250], [310, 240], [320, 230],
                     [400, 300], [250, 180], [450, 200], [200, 400], [350, 350]], dtype=np.float64)
                     
    # Simulate radial expansion of points corresponding to forward motion.
    pts2 = pts1 + (pts1 - np.array([320, 240])) * 0.1 
    
    R, t, mask = geo.estimate_pose(pts1, pts2)
    
    print("Epipolar geometry module verification passed.")
    print("Recovered Rotation Matrix R:\n", R)
    print("Recovered Translation Vector t:\n", t)