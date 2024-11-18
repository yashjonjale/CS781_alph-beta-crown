#########################################################################
##   This file is part of the auto_LiRPA library, a core part of the   ##
##   α,β-CROWN (alpha-beta-CROWN) neural network verifier developed    ##
##   by the α,β-CROWN Team                                             ##
##                                                                     ##
##   Copyright (C) 2020-2024 The α,β-CROWN Team                        ##
##   Primary contacts: Huan Zhang <huan@huan-zhang.com>                ##
##                     Zhouxing Shi <zshi@cs.ucla.edu>                 ##
##                     Kaidi Xu <kx46@drexel.edu>                      ##
##                                                                     ##
##    See CONTRIBUTORS for all author contacts and affiliations.       ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################

import json
import math
import os
import numpy as np
import torch
from .utils import logger, eyeC
from .patches import Patches, patches_to_matrix
from .linear_bound import LinearBound


class Perturbation:
    r"""
    Base class for a perturbation specification. Perturbations define the allowable
    deviations from the original input data that the network should be robust against.
    This could include adversarial attacks where inputs are slightly modified to mislead
    the network.

    **Examples:**

    - `PerturbationLpNorm`: Perturbations constrained by an Lp norm (p>=1).
    - `PerturbationL0Norm`: Perturbations constrained by the L0 norm (sparsity).
    - `PerturbationSynonym`: Synonym substitution perturbations for natural language processing tasks.
    """
    def __init__(self):
        pass

    def set_eps(self, eps):
        """
        Sets the epsilon value defining the perturbation's magnitude.

        Args:
            eps (float): The maximum allowed perturbation magnitude.
        """
        self.eps = eps

    def concretize(self, x, A, sign=-1, aux=None):
        """
        Concretizes the bounds based on the perturbation specification.

        Args:
            x (Tensor): Input tensor before perturbation.
            A (Tensor): A matrix derived from bound propagation computations.
            sign (int): Determines whether to compute lower (-1) or upper (+1) bounds.
            aux (object, optional): Auxiliary information required for concretization.

        Returns:
            Tensor: The concretized bound tensor with shape matching the clean output.
        """
        raise NotImplementedError

    def init(self, x, aux=None, forward=False):
        """
        Initializes the bounds before bound propagation computations.

        Args:
            x (Tensor): Input tensor before perturbation.
            aux (object, optional): Auxiliary information for initialization.
            forward (bool): Indicates if forward mode bound propagation is being used.

        Returns:
            Tuple:
                - LinearBound: Object containing initialized lower and upper bounds.
                - Tensor: Center tensor of perturbation.
                - object, optional: Modified or additional auxiliary information.
        """
        raise NotImplementedError


class PerturbationL0Norm(Perturbation):
    """Perturbation constrained by the L0 norm, enforcing sparsity in perturbations.

    Assumes input data values are in the range [0, 1]. The L0 norm counts the number
    of elements that are allowed to be perturbed.

    Attributes:
        eps (int): Maximum number of elements that can be perturbed.
        x_L (Tensor, optional): Lower bound tensor for inputs.
        x_U (Tensor, optional): Upper bound tensor for inputs.
        ratio (float): Scaling factor applied to the perturbation bounds.
    """

    def __init__(self, eps, x_L=None, x_U=None, ratio=1.0):
        """
        Initializes the PerturbationL0Norm instance.

        Args:
            eps (int): Maximum number of elements that can be perturbed (L0 norm).
            x_L (Tensor, optional): Lower bound for input tensor elements.
            x_U (Tensor, optional): Upper bound for input tensor elements.
            ratio (float, optional): Scaling factor for the perturbation bounds. Defaults to 1.0.
        """
        self.eps = eps
        self.x_U = x_U
        self.x_L = x_L
        self.ratio = ratio

    def concretize(self, x, A, sign=-1, aux=None):
        """
        Concretizes the bounds based on L0 norm perturbations.

        Args:
            x (Tensor): Input tensor before perturbation.
            A (Tensor): A matrix from bound propagation computations.
            sign (int): Determines if lower (-1) or upper (+1) bounds are computed.
            aux (object, optional): Auxiliary information (unused here).

        Returns:
            Tensor: The concretized bound tensor after applying L0 perturbations.
        """
        if A is None:
            return None

        # Compute the ceiling of epsilon to ensure integer number of perturbations
        eps = 1e0 * math.ceil(self.eps)
        
        # Reshape x for batch matrix multiplication
        x = x.reshape(x.shape[0], -1, 1)
        center = A.matmul(x)  # Compute the center of the bounds

        # Reshape x for element-wise operations
        x = x.reshape(x.shape[0], 1, -1)

        # Compute the original matrix multiplication result
        original = A * x.expand(x.shape[0], A.shape[-2], x.shape[2])
        
        # Create masks based on the sign of A
        neg_mask = A < 0
        pos_mask = A >= 0

        if sign == 1:
            # For upper bounds, consider positive and negative contributions
            A_diff = torch.zeros_like(A)
            A_diff[pos_mask] = A[pos_mask] - original[pos_mask]
            A_diff[neg_mask] = - original[neg_mask]
        else:
            # For lower bounds, similarly handle positive and negative contributions
            A_diff = torch.zeros_like(A)
            A_diff[pos_mask] = original[pos_mask]
            A_diff[neg_mask] = original[neg_mask] - A[pos_mask]

        # Sort the differences in descending order to select top-k perturbations
        A_diff, _ = torch.sort(A_diff, dim=2, descending=True)

        # Compute the bound by summing the top-k perturbations
        bound = center + sign * A_diff[:, :, :eps].sum(dim=2).unsqueeze(2) * self.ratio

        return bound.squeeze(2)

    def init(self, x, aux=None, forward=False):
        """
        Initializes the bounds for L0 norm perturbations.

        Args:
            x (Tensor): Input tensor before perturbation.
            aux (object, optional): Auxiliary information (unused here).
            forward (bool): Indicates if forward mode bound propagation is being used.

        Returns:
            Tuple:
                - LinearBound: Initialized bounds with lower and upper bounds.
                - Tensor: Center tensor (same as input x).
                - None: No auxiliary information returned.
        """
        x_L = x
        x_U = x
        if not forward:
            return LinearBound(None, None, None, None, x_L, x_U), x, None

        batch_size = x.shape[0]
        dim = x.reshape(batch_size, -1).shape[-1]

        # Create identity matrices for each sample in the batch
        eye = torch.eye(dim).to(x.device).unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Initialize lower and upper bounds for linear transformation
        lw = eye.reshape(batch_size, dim, *x.shape[1:])
        uw = lw.clone()
        lb = torch.zeros_like(x).to(x.device)
        ub = lb.clone()
        
        return LinearBound(lw, lb, uw, ub, x_L, x_U), x, None

    def __repr__(self):
        return 'PerturbationLpNorm(norm=0, eps={})'.format(self.eps)


class PerturbationLpNorm(Perturbation):
    """Perturbation constrained by the Lp norm.

    This class handles perturbations where the allowable deviation from the
    original input is measured using an Lp norm. It supports different norms,
    including L-infinity, L2, and others.

    Attributes:
        eps (float): Maximum allowed perturbation magnitude.
        norm (float): The order of the norm (e.g., np.inf for L-infinity).
        x_L (Tensor, optional): Lower bound tensor for inputs.
        x_U (Tensor, optional): Upper bound tensor for inputs.
        eps_min (float): Minimum epsilon value.
        dual_norm (float): Dual norm corresponding to the specified norm.
        sparse (bool): Indicates if sparse perturbations are used.
    """

    def __init__(self, eps=0, norm=np.inf, x_L=None, x_U=None, eps_min=0):
        """
        Initializes the PerturbationLpNorm instance.

        Args:
            eps (float, optional): Maximum allowed perturbation magnitude. Defaults to 0.
            norm (float, optional): The order of the norm. Defaults to np.inf (L-infinity).
            x_L (Tensor, optional): Lower bound for input tensor elements.
            x_U (Tensor, optional): Upper bound for input tensor elements.
            eps_min (float, optional): Minimum epsilon value. Defaults to 0.
        """
        self.eps = eps
        self.eps_min = eps_min
        self.norm = norm
        # Calculate the dual norm based on the specified norm
        self.dual_norm = 1 if (norm == np.inf) else (np.float64(1.0) / (1 - 1.0 / self.norm))
        self.x_L = x_L
        self.x_U = x_U
        self.sparse = False  # Indicates whether sparse perturbations are used

    def get_input_bounds(self, x, A):
        """
        Retrieves the input bounds based on the perturbation specifications.

        Args:
            x (Tensor): Input tensor before perturbation.
            A (Tensor): A matrix from bound propagation computations.

        Returns:
            Tuple[Tensor, Tensor]: Lower and upper bounds for inputs.
        """
        if self.sparse:
            # If using sparse perturbations, use precomputed sparse bounds
            if self.x_L_sparse.shape[-1] == A.shape[-1]:
                x_L, x_U = self.x_L_sparse, self.x_U_sparse
            else:
                # In backward mode, A is not sparse
                x_L, x_U = self.x_L, self.x_U
        else:
            # For standard perturbations, compute bounds based on epsilon
            x_L = x - self.eps if self.x_L is None else self.x_L
            x_U = x + self.eps if self.x_U is None else self.x_U
        return x_L, x_U

    def concretize_matrix(self, x, A, sign):
        """
        Concretizes bounds for matrix perturbations based on the Lp norm.

        Args:
            x (Tensor): Input tensor before perturbation.
            A (Tensor): A matrix from bound propagation computations.
            sign (int): Determines if lower (-1) or upper (+1) bounds are computed.

        Returns:
            Tensor: The concretized bound tensor after applying Lp perturbations.
        """
        # print(f'Concretizing matrix perturbation. Sign: {sign}')
        # print(f'x shape: {x.shape}, A shape: {A.shape}')
        # print(x)
        # print(A)
        # input()
        # If A is not an identity matrix, reshape it appropriately
        if not isinstance(A, eyeC):
            A = A.reshape(A.shape[0], A.shape[1], -1)

        if self.norm == np.inf:
            # For L-infinity norm, compute bounds based on element-wise maximum deviations
            x_L, x_U = self.get_input_bounds(x, A)
            x_ub = x_U.reshape(x_U.shape[0], -1, 1)
            x_lb = x_L.reshape(x_L.shape[0], -1, 1)
            center = (x_ub + x_lb) / 2.0  # Center of the interval
            diff = (x_ub - x_lb) / 2.0    # Half-width of the interval

            if not isinstance(A, eyeC):
                # Compute bound using the absolute values of A and the diff
                bound = A.matmul(center) + sign * A.abs().matmul(diff)
            else:
                # If A is an identity matrix, simply add or subtract the diff
                bound = center + sign * diff
        else:
            # For other norms (e.g., L2), use dual norms to compute bounds
            x = x.reshape(x.shape[0], -1, 1)
            if not isinstance(A, eyeC):
                # Compute the norm of A using the dual norm and scale by epsilon
                deviation = A.norm(self.dual_norm, -1) * self.eps
                bound = A.matmul(x) + sign * deviation.unsqueeze(-1)
            else:
                # If A is an identity matrix, add or subtract epsilon directly
                bound = x + sign * self.eps
        # Remove the last singleton dimension
        bound = bound.squeeze(-1)
        return bound

    def concretize_patches(self, x, A, sign):
        """
        Concretizes bounds for patch-based perturbations (e.g., convolutional layers).

        Args:
            x (Tensor): Input tensor before perturbation.
            A (Patches): A Patches object representing bound coefficients.
            sign (int): Determines if lower (-1) or upper (+1) bounds are computed.

        Returns:
            Tensor: The concretized bound tensor after applying patch-based perturbations.
        """
        if self.norm == np.inf:
            # For L-infinity norm, handle patch-based bounds similarly to matrix bounds
            x_L, x_U = self.get_input_bounds(x, A)

            # Compute the center and deviation of the bounds
            center = (x_U + x_L) / 2.0
            diff = (x_U - x_L) / 2.0

            if not A.identity == 1:
                # Compute bound using the absolute values of patches
                bound = A.matmul(center)
                bound_diff = A.matmul(diff, patch_abs=True)

                if sign == 1:
                    bound += bound_diff
                elif sign == -1:
                    bound -= bound_diff
                else:
                    raise ValueError("Unsupported Sign")
            else:
                # If A is an identity matrix, simply add or subtract the diff
                bound = center + sign * diff
            return bound
        else:
            # For other norms (e.g., L2), use dual norms to compute bounds
            input_shape = x.shape
            if not A.identity:
                # Convert patches to matrix form for norm computation
                matrix = patches_to_matrix(
                    A.patches, input_shape, A.stride, A.padding, A.output_shape,
                    A.unstable_idx)
                # Compute the deviation using the dual norm and epsilon
                deviation = matrix.norm(p=self.dual_norm, dim=(-3,-2,-1)) * self.eps
                # Compute the bound by adding the deviation to the center
                bound = torch.einsum('bschw,bchw->bs', matrix, x) + sign * deviation
                if A.unstable_idx is None:
                    # Reshape the bound to match output dimensions
                    bound = bound.view(matrix.size(0), A.patches.size(0),
                                       A.patches.size(2), A.patches.size(3))
            else:
                # If A is an identity matrix, simply add or subtract epsilon
                bound = x + sign * self.eps
            return bound

    def concretize(self, x, A, sign=-1, aux=None):
        """
        Concretizes bounds based on the perturbation specification.

        This method delegates to either `concretize_matrix` or `concretize_patches`
        depending on the type of the bound coefficient matrix A.

        Args:
            x (Tensor): Input tensor before perturbation.
            A (Tensor or Patches): A matrix or Patches object from bound propagation.
            sign (int): Determines if lower (-1) or upper (+1) bounds are computed.
            aux (object, optional): Auxiliary information (unused here).

        Returns:
            Tensor: The concretized bound tensor after applying perturbations.
        """
        if A is None:
            return None

        if isinstance(A, eyeC) or isinstance(A, torch.Tensor):
            # Handle matrix-based bounds
            return self.concretize_matrix(x, A, sign)
        elif isinstance(A, Patches):
            # Handle patch-based bounds (e.g., convolutional layers)
            return self.concretize_patches(x, A, sign)
        else:
            raise NotImplementedError()

    def init_sparse_linf(self, x, x_L, x_U):
        """
        Initializes sparse L-infinity norm perturbations where only a subset of dimensions
        are perturbed.

        Args:
            x (Tensor): Input tensor before perturbation.
            x_L (Tensor): Lower bound tensor for inputs.
            x_U (Tensor): Upper bound tensor for inputs.

        Returns:
            Tuple:
                - LinearBound: Object containing initialized sparse bounds.
                - Tensor: Original input tensor.
                - None: No auxiliary information returned.
        """
        self.sparse = True
        batch_size = x_L.shape[0]
        # Determine which elements can be perturbed based on their bounds
        perturbed = (x_U > x_L).int()
        logger.debug(f'Perturbed: {perturbed.sum()}')
        # Initialize lower and upper bounds for unperturbed elements
        lb = ub = x_L * (1 - perturbed)  # Elements not perturbed retain their original lower bounds
        perturbed = perturbed.view(batch_size, -1)
        # Create indices for perturbed elements
        index = torch.cumsum(perturbed, dim=-1)
        dim = max(perturbed.view(batch_size, -1).sum(dim=-1).max(), 1)
        # Initialize sparse lower and upper bounds
        self.x_L_sparse = torch.zeros(batch_size, dim + 1).to(x_L)
        self.x_L_sparse.scatter_(dim=-1, index=index, src=(x_L - lb).view(batch_size, -1), reduce='add')
        self.x_U_sparse = torch.zeros(batch_size, dim + 1).to(x_U)
        self.x_U_sparse.scatter_(dim=-1, index=index, src=(x_U - ub).view(batch_size, -1), reduce='add')
        # Remove the first column as it's used for indexing
        self.x_L_sparse, self.x_U_sparse = self.x_L_sparse[:, 1:], self.x_U_sparse[:, 1:]
        
        # Initialize lower and upper weight matrices for sparse perturbations
        lw = torch.zeros(batch_size, dim + 1, perturbed.shape[-1], device=x.device)
        perturbed = perturbed.to(torch.get_default_dtype())
        lw.scatter_(dim=1, index=index.unsqueeze(1), src=perturbed.unsqueeze(1))
        lw = uw = lw[:, 1:, :].view(batch_size, dim, *x.shape[1:])
        
        print(f'Using Linf sparse perturbation. Perturbed dimensions: {dim}.')
        print(f'Avg perturbation: {(self.x_U_sparse - self.x_L_sparse).mean()}')
        
        return LinearBound(
            lw, lb, uw, ub, x_L, x_U), x, None

    def init(self, x, aux=None, forward=False):
        """
        Initializes the bounds based on the specified norm and perturbation type.

        Args:
            x (Tensor): Input tensor before perturbation.
            aux (object, optional): Auxiliary information for initialization.
            forward (bool): Indicates if forward mode bound propagation is being used.

        Returns:
            Tuple:
                - LinearBound: Object containing initialized lower and upper bounds.
                - Tensor: Center tensor of perturbation.
                - object, optional: Auxiliary information (unused here).
        """
        self.sparse = False
        if self.norm == np.inf:
            # For L-infinity norm, compute bounds by adding and subtracting epsilon
            x_L = x - self.eps if self.x_L is None else self.x_L
            x_U = x + self.eps if self.x_U is None else self.x_U
        else:
            if int(os.environ.get('AUTOLIRPA_L2_DEBUG', 0)) == 1:
                # Experimental code for debugging L2 norm perturbations
                x_L = x - self.eps if self.x_L is None else self.x_L
                x_U = x + self.eps if self.x_U is None else self.x_U
            else:
                # For other norms, use the original tensor for both lower and upper bounds
                x_L = x_U = x

        if not forward:
            # If not in forward mode, initialize bounds without transformation
            return LinearBound(
                None, None, None, None, x_L, x_U), x, None

        if (self.norm == np.inf and x_L.numel() > 1
                and (x_L == x_U).sum() > 0.5 * x_L.numel()):
            # If using L-infinity norm and many elements are unperturbed, use sparse perturbations
            return self.init_sparse_linf(x, x_L, x_U)

        batch_size = x.shape[0]
        dim = x.reshape(batch_size, -1).shape[-1]

        # Initialize lower and upper bias tensors
        lb = ub = torch.zeros_like(x)
        # Create identity matrices for each sample in the batch
        eye = torch.eye(dim).to(x).expand(batch_size, dim, dim)
        lw = uw = eye.reshape(batch_size, dim, *x.shape[1:])
        
        return LinearBound(
            lw, lb, uw, ub, x_L, x_U), x, None

    def __repr__(self):
        """
        Returns a string representation of the PerturbationLpNorm instance.

        Returns:
            str: String representation.
        """
        if self.norm == np.inf:
            if self.x_L is None and self.x_U is None:
                return f'PerturbationLpNorm(norm=inf, eps={self.eps},)'
            else:
                return f'PerturbationLpNorm(norm=inf, eps={self.eps}, x_L={self.x_L}, x_U={self.x_U})'
        else:
            return f'PerturbationLpNorm(norm={self.norm}, eps={self.eps})'



class PerturbationSynonym(Perturbation):
    """
    Perturbation based on synonym substitution, primarily used in Natural Language Processing (NLP) tasks.

    This perturbation allows for replacing words in the input with their synonyms within a specified
    substitution budget, enhancing the robustness of NLP models against synonym-based adversarial attacks.

    Attributes:
        budget (int): Maximum number of words that can be substituted.
        eps (float): Scaling factor for perturbations.
        use_simple (bool): Flag to use a simplified substitution strategy.
        model (object, optional): Language model or synonym model used for substitutions.
        train (bool): Indicates if the model is in training mode.
        synonym (dict): Dictionary mapping words to their possible synonyms.
    """

    def __init__(self, budget, eps=1.0, use_simple=False):
        """
        Initializes the PerturbationSynonym instance.

        Args:
            budget (int): Maximum number of words that can be substituted.
            eps (float, optional): Scaling factor for perturbations. Defaults to 1.0.
            use_simple (bool, optional): Flag to use a simplified substitution strategy. Defaults to False.
        """
        super(PerturbationSynonym, self).__init__()
        self.budget = budget
        self.eps = eps
        self.use_simple = use_simple
        self.model = None  # Placeholder for a language model or synonym model
        self.train = False  # Indicates if the model is in training mode

    def __repr__(self):
        return (f'perturbation(Synonym-based word substitution '
                f'budget={self.budget}, eps={self.eps})')

    def _load_synonyms(self, path='data/synonyms.json'):
        """
        Loads synonyms from a JSON file into a dictionary.

        Args:
            path (str, optional): Path to the synonyms JSON file. Defaults to 'data/synonyms.json'.
        """
        with open(path) as file:
            self.synonym = json.loads(file.read())
        logger.info('Synonym list loaded for {} words'.format(len(self.synonym)))

    def set_train(self, train):
        """
        Sets the training mode for the synonym model.

        Args:
            train (bool): If True, sets the model to training mode; otherwise, evaluation mode.
        """
        self.train = train

    def concretize(self, x, A, sign, aux):
        """
        Concretizes bounds based on synonym substitutions.

        This involves replacing words with their synonyms within the allowed budget and
        computing the resulting bounds.

        Args:
            x (Tensor): Input tensor before perturbation.
            A (Tensor): A matrix from bound propagation computations.
            sign (int): Determines if lower (-1) or upper (+1) bounds are computed.
            aux (tuple): Auxiliary information containing representation of words, masks, etc.

        Returns:
            Tensor: The concretized bound tensor after applying synonym substitutions.
        """
        assert(self.model is not None)

        # Unpack auxiliary information
        x_rep, mask, can_be_replaced = aux
        batch_size, length, dim_word = x.shape[0], x.shape[1], x.shape[2]
        dim_out = A.shape[1]
        max_num_cand = x_rep.shape[2]

        # Convert replacement mask to tensor
        mask_rep = torch.tensor(can_be_replaced, dtype=torch.get_default_dtype(), device=A.device)

        # Determine the maximum number of substitutions allowed per sample
        num_pos = int(np.max(np.sum(can_be_replaced, axis=-1)))
        update_A = A.shape[-1] > num_pos * dim_word
        if update_A:
            # Compute bias based on unperturbed elements
            bias = torch.bmm(A, (x * (1 - mask_rep).unsqueeze(-1)).reshape(batch_size, -1, 1)).squeeze(-1)
        else:
            bias = 0.
        # Reshape A for matrix multiplication
        A = A.reshape(batch_size, dim_out, -1, dim_word)

        # Initialize lists to store updated bounds and substitutions
        A_new, x_new, x_rep_new, mask_new = [], [], [], []
        zeros_A = torch.zeros(dim_out, dim_word, device=A.device)
        zeros_w = torch.zeros(dim_word, device=A.device)
        zeros_rep = torch.zeros(max_num_cand, dim_word, device=A.device)
        zeros_mask = torch.zeros(max_num_cand, device=A.device)
        for t in range(batch_size):
            cnt = 0
            for i in range(0, length):
                if can_be_replaced[t][i]:
                    if update_A:
                        # Append original A matrix for unperturbed words
                        A_new.append(A[t, :, i, :])
                    # Append original word vector
                    x_new.append(x[t][i])
                    # Append representation for substitutions
                    x_rep_new.append(x_rep[t][i])
                    # Append mask indicating substitutions
                    mask_new.append(mask[t][i])
                    cnt += 1
            if update_A:
                # Pad with zeros if fewer substitutions than budget
                A_new += [zeros_A] * (num_pos - cnt)
            # Pad with zeros for word vectors and substitutions
            x_new += [zeros_w] * (num_pos - cnt)
            x_rep_new += [zeros_rep] * (num_pos - cnt)
            mask_new += [zeros_mask] * (num_pos - cnt)
        if update_A:
            # Reshape and transpose A for batch processing
            A = torch.cat(A_new).reshape(batch_size, num_pos, dim_out, dim_word).transpose(1, 2)
        # Concatenate and reshape word vectors and substitution representations
        x_new = torch.cat(x_new).reshape(batch_size, num_pos, dim_word)
        x_rep_new = torch.cat(x_rep_new).reshape(batch_size, num_pos, max_num_cand, dim_word)
        mask_new = torch.cat(mask_new).reshape(batch_size, num_pos, max_num_cand)
        length = num_pos

        # Reshape A for matrix multiplication
        A = A.reshape(batch_size, A.shape[1], length, -1).transpose(1, 2)
        x = x_new.reshape(batch_size, length, -1, 1)

        if sign == 1:
            cmp, init = torch.max, -1e30  # For upper bounds, use max
        else:
            cmp, init = torch.min, 1e30   # For lower bounds, use min

        # Initialize tensor to store bound values
        init_tensor = torch.ones(batch_size, dim_out).to(x.device) * init
        dp = [[init_tensor] * (self.budget + 1) for i in range(0, length + 1)]
        dp[0][0] = torch.zeros(batch_size, dim_out).to(x.device)

        # Reshape A for batch matrix multiplication
        A = A.reshape(batch_size * length, A.shape[2], A.shape[3])
        Ax = torch.bmm(
            A,
            x.reshape(batch_size * length, x.shape[2], x.shape[3])
        ).reshape(batch_size, length, A.shape[1])

        # Compute bounds for substitutions
        Ax_rep = torch.bmm(
            A,
            x_rep.reshape(batch_size * length, max_num_cand, x.shape[2]).transpose(-1, -2)
        ).reshape(batch_size, length, A.shape[1], max_num_cand)
        Ax_rep = Ax_rep * mask.unsqueeze(2) + init * (1 - mask).unsqueeze(2)
        Ax_rep_bound = cmp(Ax_rep, dim=-1).values

        if self.use_simple and self.train:
            # Simplified bound computation during training
            return torch.sum(cmp(Ax, Ax_rep_bound), dim=1) + bias

        # Dynamic Programming approach to handle budgeted substitutions
        for i in range(1, length + 1):
            dp[i][0] = dp[i - 1][0] + Ax[:, i - 1]
            for j in range(1, self.budget + 1):
                dp[i][j] = cmp(
                    dp[i - 1][j] + Ax[:, i - 1],
                    dp[i - 1][j - 1] + Ax_rep_bound[:, i - 1]
                )
        dp = torch.cat(dp[length], dim=0).reshape(self.budget + 1, batch_size, dim_out)

        # Return the final bound based on the maximum/minimum over the budget
        return cmp(dp, dim=0).values + bias

    def init(self, x, aux=None, forward=False):
        """
        Initializes the bounds for synonym-based perturbations.

        Args:
            x (Tensor): Input tensor before perturbation.
            aux (object, optional): Auxiliary information containing token information.
            forward (bool): Indicates if forward mode bound propagation is being used.

        Returns:
            Tuple:
                - LinearBound: Initialized bounds with lower and upper bounds.
                - Tensor: Original input tensor.
                - Tuple: Auxiliary information for substitution representations.
        """
        tokens, batch = aux  # Unpack auxiliary information containing tokens and batch data
        self.tokens = tokens  # DEBUG: Store tokens for reference
        assert(len(x.shape) == 3)  # Expecting input tensor to have 3 dimensions (batch, length, embedding_dim)

        batch_size, length, dim_word = x.shape[0], x.shape[1], x.shape[2]
        max_pos = 1
        # Initialize a mask indicating which words can be replaced
        can_be_replaced = np.zeros((batch_size, length), dtype=bool)

        # Build substitution candidates for each example in the batch
        self._build_substitution(batch)

        for t in range(batch_size):
            cnt = 0
            candidates = batch[t]['candidates']
            # Handle special tokens for transformer models
            if tokens[t][0] == '[CLS]':
                candidates = [[]] + candidates + [[]]
            for i in range(len(tokens[t])):
                # Check if the word can be replaced based on candidates
                if tokens[t][i] == '[UNK]' or \
                        len(candidates[i]) == 0 or tokens[t][i] != candidates[i][0]:
                    continue
                for w in candidates[i][1:]:
                    if w in self.model.vocab:
                        # Mark the word as replaceable and limit to one substitution per word
                        can_be_replaced[t][i] = True
                        cnt += 1
                        break
            # Update the maximum number of substitutions across the batch
            max_pos = max(max_pos, cnt)

        dim = max_pos * dim_word  # Total dimension accounting for substitutions
        if forward:
            # Initialize identity matrices for linear bounds in forward mode
            eye = torch.eye(dim_word).to(x.device)
            lw = torch.zeros(batch_size, dim, length, dim_word).to(x.device)
            lb = torch.zeros_like(x).to(x.device)
        word_embeddings = self.model.word_embeddings.weight  # Word embedding matrix from the model
        vocab = self.model.vocab  # Vocabulary mapping

        # Initialize lists to store substitution representations and masks
        x_rep = [[[] for i in range(length)] for t in range(batch_size)]
        max_num_cand = 1  # Maximum number of substitution candidates

        for t in range(batch_size):
            candidates = batch[t]['candidates']
            # Handle special tokens for transformer models
            if tokens[t][0] == '[CLS]':
                candidates = [[]] + candidates + [[]]
            cnt = 0
            for i in range(length):
                if can_be_replaced[t][i]:
                    # Extract word embedding and compute residual (difference from embedding)
                    word_embed = word_embeddings[vocab[tokens[t][i]]]
                    other_embed = x[t, i] - word_embed
                    if forward:
                        # Assign identity to the substitution weights
                        lw[t, (cnt * dim_word):((cnt + 1) * dim_word), i, :] = eye
                        lb[t, i, :] = torch.zeros_like(word_embed)
                    for w in candidates[i][1:]:
                        if w in self.model.vocab:
                            # Append the substituted word embedding plus residual
                            x_rep[t][i].append(
                                word_embeddings[self.model.vocab[w]] + other_embed)
                    # Update the maximum number of substitution candidates
                    max_num_cand = max(max_num_cand, len(x_rep[t][i]))
                    cnt += 1
                else:
                    if forward:
                        # Assign original word embedding to the lower bound if not substitutable
                        lb[t, i, :] = x[t, i, :]
        
        if forward:
            # Assign upper bounds similarly
            uw, ub = lw.clone(), lb.clone()
        else:
            # If not in forward mode, no bounds are assigned
            lw = lb = uw = ub = None

        # Initialize a zero tensor for non-substituted words
        zeros = torch.zeros(dim_word, device=x.device)

        # Prepare substitution representations and masks
        x_rep_, mask = [], []
        for t in range(batch_size):
            for i in range(length):
                # Append substitution embeddings and pad with zeros if necessary
                x_rep_ += x_rep[t][i] + [zeros] * (max_num_cand - len(x_rep[t][i]))
                mask += [1] * len(x_rep[t][i]) + [0] * (max_num_cand - len(x_rep[t][i]))
        # Reshape substitution representations and masks
        x_rep_ = torch.cat(x_rep_).reshape(batch_size, length, max_num_cand, dim_word)
        mask = torch.tensor(mask, dtype=torch.get_default_dtype(), device=x.device)\
            .reshape(batch_size, length, max_num_cand)
        # Apply scaling to substitution embeddings
        x_rep_ = x_rep_ * self.eps + x.unsqueeze(2) * (1 - self.eps)

        inf = 1e20  # Define a large constant for initialization
        # Compute lower bounds by selecting the minimum substitution values
        lower = torch.min(mask.unsqueeze(-1) * x_rep_ + (1 - mask).unsqueeze(-1) * inf, dim=2).values
        # Compute upper bounds by selecting the maximum substitution values
        upper = torch.max(mask.unsqueeze(-1) * x_rep_ + (1 - mask).unsqueeze(-1) * (-inf), dim=2).values
        # Ensure that the bounds include the original input values
        lower = torch.min(lower, x)
        upper = torch.max(upper, x)

        return LinearBound(lw, lb, uw, ub, lower, upper), x, (x_rep_, mask, can_be_replaced)

    def _build_substitution(self, batch):
        """
        Builds substitution candidates for each example in the batch.

        Args:
            batch (list): List of examples containing sentences and candidate substitutions.
        """
        for example in batch:
            if not 'candidates' in example or example['candidates'] is None:
                candidates = []
                tokens = example['sentence'].strip().lower().split(' ')
                for i in range(len(tokens)):
                    _cand = []
                    if tokens[i] in self.synonym:
                        for w in self.synonym[tokens[i]]:
                            if w in self.model.vocab:
                                _cand.append(w)
                    if len(_cand) > 0:
                        # Prepend the original word to the list of candidates
                        _cand = [tokens[i]] + _cand
                    candidates.append(_cand)
                example['candidates'] = candidates



class PerturbationLpNormLocalised(Perturbation):
    """Perturbation constrained by the Lp norm.

    This class handles perturbations where the allowable deviation from the
    original input is measured using an Lp norm. It supports different norms,
    including L-infinity, L2, and others.

    Attributes:
        eps (float): Maximum allowed perturbation magnitude.
        norm (float): The order of the norm (e.g., np.inf for L-infinity).
        x_L (Tensor, optional): Lower bound tensor for inputs.
        x_U (Tensor, optional): Upper bound tensor for inputs.
        eps_min (float): Minimum epsilon value.
        dual_norm (float): Dual norm corresponding to the specified norm.
        sparse (bool): Indicates if sparse perturbations are used.
    """

    def __init__(self, window_size, eps=0, norm=np.inf, x_L=None, x_U=None, eps_min=0): ### add the window size as the argument
        """
        Initializes the PerturbationLpNorm instance.

        Args:
            eps (float, optional): Maximum allowed perturbation magnitude. Defaults to 0.
            norm (float, optional): The order of the norm. Defaults to np.inf (L-infinity).
            x_L (Tensor, optional): Lower bound for input tensor elements.
            x_U (Tensor, optional): Upper bound for input tensor elements.
            eps_min (float, optional): Minimum epsilon value. Defaults to 0.
        """
        self.eps = eps
        self.eps_min = eps_min
        self.norm = norm
        # Calculate the dual norm based on the specified norm
        self.dual_norm = 1 if (norm == np.inf) else (np.float64(1.0) / (1 - 1.0 / self.norm))
        self.x_L = x_L
        self.x_U = x_U
        self.sparse = False  # Indicates whether sparse perturbations are used
        self.window_size = window_size

    def get_input_bounds(self, x, A):
        """
        Retrieves the input bounds based on the perturbation specifications.

        Args:
            x (Tensor): Input tensor before perturbation.
            A (Tensor): A matrix from bound propagation computations.

        Returns:
            Tuple[Tensor, Tensor]: Lower and upper bounds for inputs.
        """
        if self.sparse:
            # If using sparse perturbations, use precomputed sparse bounds
            if self.x_L_sparse.shape[-1] == A.shape[-1]:
                x_L, x_U = self.x_L_sparse, self.x_U_sparse
            else:
                # In backward mode, A is not sparse
                x_L, x_U = self.x_L, self.x_U
        else:
            # For standard perturbations, compute bounds based on epsilon
            x_L = x - self.eps if self.x_L is None else self.x_L
            x_U = x + self.eps if self.x_U is None else self.x_U
        return x_L, x_U
    
    @staticmethod
    def get_deviation(A: torch.Tensor,c:int, I: int, k: int, p: int):
        """
        A - shape (B, O, Chan*I*I)
        """
        # print("A shape", A.shape)
        # print(c, I, k, p)
        B, O, _ = A.shape
        Avw = (torch.abs(A)**p).reshape(B*O, 1, c, I, I)
        kernel = torch.ones(1, 1, c, k, k).to(A.device)
        out = torch.nn.functional.conv3d(Avw, kernel)
        out = out**(1/p)
        out = out.view(B, O, -1).max(dim=-1).values
        return out


    def concretize_matrix(self, x, A, sign):
        """
        Concretizes bounds for matrix perturbations based on the Lp norm.

        Args:
            x (Tensor): Input tensor before perturbation.
            A (Tensor): A matrix from bound propagation computations.
            sign (int): Determines if lower (-1) or upper (+1) bounds are computed.

            x - input tensor with shape - (batch_size, channels, I,I)
            A - matrix with shape - (batch_size, O, (channels*I*I))

        Returns:
            Tensor: The concretized bound tensor after applying Lp perturbations.
        """
        # If A is not an identity matrix, reshape it appropriately
        # print("A shape", A.shape)
        if not isinstance(A, eyeC):
            A = A.reshape(A.shape[0], A.shape[1], -1) # (batch_size, O, (I*I))

        # if self.norm == np.inf:
        #     # For L-infinity norm, compute bounds based on element-wise maximum deviations
        #     x_L, x_U = self.get_input_bounds(x, A)
        #     x_ub = x_U.reshape(x_U.shape[0], -1, 1)
        #     x_lb = x_L.reshape(x_L.shape[0], -1, 1)
        #     center = (x_ub + x_lb) / 2.0  # Center of the interval
        #     diff = (x_ub - x_lb) / 2.0    # Half-width of the interval

        #     if not isinstance(A, eyeC):
        #         # Compute bound using the absolute values of A and the diff
        #         bound = A.matmul(center) + sign * A.abs().matmul(diff) ##basically the dot product calculation
        #     else:
        #         # If A is an identity matrix, simply add or subtract the diff
        #         bound = center + sign * diff
        # else:
        #     # For other norms (e.g., L2), use dual norms to compute bounds
        channels, I, _ = x.shape[1], x.shape[2], x.shape[3]
        x = x.reshape(x.shape[0], -1, 1) # (batch_size, channels*I*I, 1)
        # if not isinstance(A, eyeC):
            # Compute the norm of A using the dual norm and scale by epsilon
        deviation = self.get_deviation(A,channels,I,self.window_size,self.dual_norm) * self.eps
        
        bound = A.matmul(x) + sign * deviation.unsqueeze(-1)
        # else:
            # If A is an identity matrix, add or subtract epsilon directly
            # bound = x + sign * self.eps
        # Remove the last singleton dimension
        bound = bound.squeeze(-1)
        return bound

    def concretize_patches(self, x, A, sign):
        """
        Concretizes bounds for patch-based perturbations (e.g., convolutional layers).

        Args:
            x (Tensor): Input tensor before perturbation.
            A (Patches): A Patches object representing bound coefficients.
            sign (int): Determines if lower (-1) or upper (+1) bounds are computed.

        Returns:
            Tensor: The concretized bound tensor after applying patch-based perturbations.
        """
        if self.norm == np.inf:
            # For L-infinity norm, handle patch-based bounds similarly to matrix bounds
            x_L, x_U = self.get_input_bounds(x, A)

            # Compute the center and deviation of the bounds
            center = (x_U + x_L) / 2.0
            diff = (x_U - x_L) / 2.0

            if not A.identity == 1:
                # Compute bound using the absolute values of patches
                bound = A.matmul(center)
                bound_diff = A.matmul(diff, patch_abs=True)

                if sign == 1:
                    bound += bound_diff
                elif sign == -1:
                    bound -= bound_diff
                else:
                    raise ValueError("Unsupported Sign")
            else:
                # If A is an identity matrix, simply add or subtract the diff
                bound = center + sign * diff
            return bound
        else:
            # For other norms (e.g., L2), use dual norms to compute bounds
            input_shape = x.shape
            if not A.identity:
                # Convert patches to matrix form for norm computation
                matrix = patches_to_matrix(
                    A.patches, input_shape, A.stride, A.padding, A.output_shape,
                    A.unstable_idx)
                # Compute the deviation using the dual norm and epsilon
                deviation = matrix.norm(p=self.dual_norm, dim=(-3,-2,-1)) * self.eps
                # Compute the bound by adding the deviation to the center
                bound = torch.einsum('bschw,bchw->bs', matrix, x) + sign * deviation
                if A.unstable_idx is None:
                    # Reshape the bound to match output dimensions
                    bound = bound.view(matrix.size(0), A.patches.size(0),
                                       A.patches.size(2), A.patches.size(3))
            else:
                # If A is an identity matrix, simply add or subtract epsilon
                bound = x + sign * self.eps
            return bound

    def concretize(self, x, A, sign=-1, aux=None):
        """
        Concretizes bounds based on the perturbation specification.

        This method delegates to either `concretize_matrix` or `concretize_patches`
        depending on the type of the bound coefficient matrix A.

        Args:
            x (Tensor): Input tensor before perturbation.
            A (Tensor or Patches): A matrix or Patches object from bound propagation.
            sign (int): Determines if lower (-1) or upper (+1) bounds are computed.
            aux (object, optional): Auxiliary information (unused here).

        Returns:
            Tensor: The concretized bound tensor after applying perturbations.
        """
        if A is None:
            return None
        if isinstance(A, eyeC) or isinstance(A, torch.Tensor):
            # Handle matrix-based bounds
            return self.concretize_matrix(x, A, sign)
        elif isinstance(A, Patches):
            # Handle patch-based bounds (e.g., convolutional layers)
            return self.concretize_patches(x, A, sign)
        else:
            raise NotImplementedError()

    def init_sparse_linf(self, x, x_L, x_U):
        """
        Initializes sparse L-infinity norm perturbations where only a subset of dimensions
        are perturbed.

        Args:
            x (Tensor): Input tensor before perturbation.
            x_L (Tensor): Lower bound tensor for inputs.
            x_U (Tensor): Upper bound tensor for inputs.

        Returns:
            Tuple:
                - LinearBound: Object containing initialized sparse bounds.
                - Tensor: Original input tensor.
                - None: No auxiliary information returned.
        """
        self.sparse = True
        batch_size = x_L.shape[0]
        # Determine which elements can be perturbed based on their bounds
        perturbed = (x_U > x_L).int()
        logger.debug(f'Perturbed: {perturbed.sum()}')
        # Initialize lower and upper bounds for unperturbed elements
        lb = ub = x_L * (1 - perturbed)  # Elements not perturbed retain their original lower bounds
        perturbed = perturbed.view(batch_size, -1)
        # Create indices for perturbed elements
        index = torch.cumsum(perturbed, dim=-1)
        dim = max(perturbed.view(batch_size, -1).sum(dim=-1).max(), 1)
        # Initialize sparse lower and upper bounds
        self.x_L_sparse = torch.zeros(batch_size, dim + 1).to(x_L)
        self.x_L_sparse.scatter_(dim=-1, index=index, src=(x_L - lb).view(batch_size, -1), reduce='add')
        self.x_U_sparse = torch.zeros(batch_size, dim + 1).to(x_U)
        self.x_U_sparse.scatter_(dim=-1, index=index, src=(x_U - ub).view(batch_size, -1), reduce='add')
        # Remove the first column as it's used for indexing
        self.x_L_sparse, self.x_U_sparse = self.x_L_sparse[:, 1:], self.x_U_sparse[:, 1:]
        
        # Initialize lower and upper weight matrices for sparse perturbations
        lw = torch.zeros(batch_size, dim + 1, perturbed.shape[-1], device=x.device)
        perturbed = perturbed.to(torch.get_default_dtype())
        lw.scatter_(dim=1, index=index.unsqueeze(1), src=perturbed.unsqueeze(1))
        lw = uw = lw[:, 1:, :].view(batch_size, dim, *x.shape[1:])
        
        print(f'Using Linf sparse perturbation. Perturbed dimensions: {dim}.')
        print(f'Avg perturbation: {(self.x_U_sparse - self.x_L_sparse).mean()}')
        
        return LinearBound(
            lw, lb, uw, ub, x_L, x_U), x, None

    def init(self, x, aux=None, forward=False):
        """
        Initializes the bounds based on the specified norm and perturbation type.

        Args:
            x (Tensor): Input tensor before perturbation.
            aux (object, optional): Auxiliary information for initialization.
            forward (bool): Indicates if forward mode bound propagation is being used.

        Returns:
            Tuple:
                - LinearBound: Object containing initialized lower and upper bounds.
                - Tensor: Center tensor of perturbation.
                - object, optional: Auxiliary information (unused here).
        """
        self.sparse = False
        if self.norm == np.inf:
            # For L-infinity norm, compute bounds by adding and subtracting epsilon
            x_L = x - self.eps if self.x_L is None else self.x_L
            x_U = x + self.eps if self.x_U is None else self.x_U
        else:
            if int(os.environ.get('AUTOLIRPA_L2_DEBUG', 0)) == 1:
                # Experimental code for debugging L2 norm perturbations
                x_L = x - self.eps if self.x_L is None else self.x_L
                x_U = x + self.eps if self.x_U is None else self.x_U
            else:
                # For other norms, use the original tensor for both lower and upper bounds
                x_L = x_U = x

        if not forward:
            # If not in forward mode, initialize bounds without transformation
            return LinearBound(
                None, None, None, None, x_L, x_U), x, None

        if (self.norm == np.inf and x_L.numel() > 1
                and (x_L == x_U).sum() > 0.5 * x_L.numel()):
            # If using L-infinity norm and many elements are unperturbed, use sparse perturbations
            return self.init_sparse_linf(x, x_L, x_U)

        batch_size = x.shape[0]
        dim = x.reshape(batch_size, -1).shape[-1]

        # Initialize lower and upper bias tensors
        lb = ub = torch.zeros_like(x)
        # Create identity matrices for each sample in the batch
        eye = torch.eye(dim).to(x).expand(batch_size, dim, dim)
        lw = uw = eye.reshape(batch_size, dim, *x.shape[1:])
        
        return LinearBound(
            lw, lb, uw, ub, x_L, x_U), x, None

    def __repr__(self):
        """
        Returns a string representation of the PerturbationLpNorm instance.

        Returns:
            str: String representation.
        """
        if self.norm == np.inf:
            if self.x_L is None and self.x_U is None:
                return f'PerturbationLpNorm(norm=inf, eps={self.eps}, window size={self.window_size})'
            else:
                return f'PerturbationLpNorm(norm=inf, eps={self.eps}, window_size={self.window_size}, x_L={self.x_L}, x_U={self.x_U})'
        else:
            return f'PerturbationLpNorm(norm={self.norm}, eps={self.eps}, window_size={self.window_size})'