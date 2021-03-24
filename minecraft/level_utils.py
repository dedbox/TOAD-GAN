from typing import List
from config import Config
import torch
import numpy as np
import os
import shutil
from loguru import logger
import torch.nn.functional as F

# import minecraft.nbt as nbt
from PyAnvilEditor.pyanvil import World, BlockState, Canvas
from utils import load_pkl


# Miscellaneous functions to deal with MC schematics.

def load_schematic(path_to_schem):
    """ Loads a Minecraft .schem file """
    '''
    sch = nbt.load(path_to_schem)
    # _Blocks is y, z, x
    blocks = sch["Blocks"].value.astype('uint16').reshape(sch["Height"].value, sch["Length"].value, sch["Width"].value)
    data = sch["Data"].value.reshape(sch["Height"].value, sch["Length"].value, sch["Width"].value)

    blockdata = np.concatenate([blocks.reshape(blocks.shape + (1,)), data.reshape(data.shape + (1,))], axis=3)

    return blockdata
    '''


class NanoMCSchematic:
    def __init__(self, filename, shape, mats='Alpha'):
        """ We assume shape is already in yzx, unlike original code! """
        '''
        self.filename = filename
        self.material = mats
        assert shape is not None
        root_tag = nbt.TAG_Compound(name="Schematic")
        #
        root_tag["Height"] = nbt.TAG_Short(shape[0])
        root_tag["Length"] = nbt.TAG_Short(shape[1])
        root_tag["Width"] = nbt.TAG_Short(shape[2])

        root_tag["Entities"] = nbt.TAG_List()
        root_tag["TileEntities"] = nbt.TAG_List()
        root_tag["TileTicks"] = nbt.TAG_List()
        root_tag["Materials"] = nbt.TAG_String(self.material)

        self._Blocks = np.zeros((shape[0], shape[1], shape[2]), 'uint16')
        self._Data = np.zeros((shape[0], shape[1], shape[2]), 'uint16')
        # root_tag["Data"] = nbt.TAG_Byte_Array(np.zeros((shape[1], shape[2], shape[0]), np.uint8))

        root_tag["Biomes"] = nbt.TAG_Byte_Array(np.zeros((shape[2], shape[0]), np.uint8))

        self.root_tag = root_tag

    def set_blockdata(self, blockdata):
        blocks = blockdata[:, :, :, 0]
        data = blockdata[:, :, :, 1]
        assert blocks.shape == self._Blocks.shape
        assert data.shape == self._Data.shape
        self._Blocks = blocks.astype(type(self._Blocks))
        self._Data = data.astype(type(self._Data))

    def saveToFile(self, filename=None):
        """ save to file named filename, or use self.filename.  XXX NOT THREAD SAFE AT ALL. """
        if filename is None:
            filename = self.filename
        if filename is None:
            raise IOError("Attempted to save an unnamed schematic in place")

        self.Materials = self.material

        self.root_tag["Blocks"] = nbt.TAG_Byte_Array(self._Blocks.astype('uint8'))
        self.root_tag["Data"] = nbt.TAG_Byte_Array(self._Data.astype('uint8'))
        self.root_tag["Data"].value &= 0xF  # discard high bits

        add = self._Blocks >> 8
        if add.any():
            # WorldEdit AddBlocks compatibility.
            # The first 4-bit value is stored in the high bits of the first byte.

            # Increase odd size by one to align slices.
            packed_add = np.zeros(add.size + (add.size & 1), 'uint8')
            packed_add[:add.size] = add.ravel()

            # Shift even bytes to the left
            packed_add[::2] <<= 4

            # Merge odd bytes into even bytes
            packed_add[::2] |= packed_add[1::2]

            # Save only the even bytes, now that they contain the odd bytes in their lower bits.
            packed_add = packed_add[0::2]
            self.root_tag["AddBlocks"] = nbt.TAG_Byte_Array(packed_add)

        with open(filename, 'wb') as chunkfh:
            self.root_tag.save(chunkfh)
            chunkfh.close()

        del self.root_tag["Blocks"]
        del self.root_tag["Data"]
        self.root_tag.pop("AddBlocks", None)
        '''


# def blockdata_to_one_hot_level(bdata, tokens) -> torch.Tensor:
#     """ Converts blockdata to a full token level tensor. """
#     oh_level = torch.zeros((len(tokens),) + bdata.shape[:-1])
#     for i, tok in enumerate(tokens):
#         overlap = bdata[:, :, :] == tok
#         true_pos = np.logical_and(overlap[:, :, :, 0], overlap[:, :, :, 1])
#         oh_level[i][true_pos] = 1
#
#     return oh_level


def one_hot_to_blockdata_level(oh_level, tokens, block2repr, repr_type):
    """ Converts a full token level tensor to blockdata. """
    # Representations
    # block2repr = load_pkl('prim_cutout_representations', prepath='/home/awiszus/Project/TOAD-GAN/input/minecraft/')

    if repr_type == "autoencoder":
        oh_level = block2repr["decoder"](oh_level).detach()

    bdata = np.zeros(oh_level.shape[2:], 'uint8')
    for y in range(bdata.shape[0]):
        for z in range(bdata.shape[1]):
            for x in range(bdata.shape[2]):
                if repr_type == "block2vec":
                    dists = np.zeros((len(tokens),))
                    for i, rep in enumerate(tokens):
                        if isinstance(block2repr[rep], list):
                            dists2 = np.zeros((len(block2repr[rep]),))
                            for j, rep2 in enumerate(block2repr[rep]):
                                dists2[j] = F.mse_loss(rep2, oh_level[0, :, y, z, x].detach().cpu()).detach()
                            dists[i] = dists2.min()
                        else:
                            dists[i] = F.mse_loss(block2repr[rep], oh_level[0, :, y, z, x].detach().cpu()).detach()
                    bdata[y, z, x] = dists.argmin()
                else:
                    bdata[y, z, x] = oh_level[:, :, y, z, x].argmax()

    return bdata


# def read_level_from_file(input_dir, input_name):
#     """ Returns a full token level tensor from a .txt file. Also returns the unique tokens found in this level.
#     Token. """
#     bdata = load_schematic(os.path.join(input_dir, input_name))
#     uniques = set()
#     for y in range(bdata.shape[0]):
#         for z in range(bdata.shape[1]):
#             for x in range(bdata.shape[2]):
#                 uniques.add(tuple(bdata[y, z, x]))
#     uniques = list(uniques)
#     uniques.sort()  # necessary! otherwise we won't know the token order later
#     oh_level = blockdata_to_one_hot_level(bdata, uniques)
#     return oh_level.unsqueeze(dim=0), uniques


def read_level(opt: Config):
    """ Wrapper function for read_level_from_file using namespace opt. Updates parameters for opt."""
    # If we have multiple levels as input, we need to sync the tokens

    # Default: Only one input level
    # with World Files, we need the coords of our actual level
    if not opt.coords:
        # Ruins
        # opt.coords = ((1028, 1076), (60, 80), (1088, 1127))  # y, z, x
        opt.coords = ((1044, 1060), (64, 80), (1104, 1120))  # y, z, x
        # Fortress
        # opt.coords = ((-1735, -1673), (146, 182), (1861, 1902))  # y, z, x
        # opt.coords = ((-1726, -1709), (146, 182), (1891, 1904))  # y, z, x

    level, uniques, props = read_level_from_file(opt.input_dir, opt.input_name, opt.coords,
                                                 opt.block2repr, opt.repr_type)
    opt.token_list = uniques
    opt.props = props
    logger.info("Tokens in level {}", opt.token_list)
    opt.nc_current = level.shape[1]
    return level


def read_level_from_file(input_dir, input_name, coords, block2repr, repr_type, debug=False):
    """ coords is ((y0,yend), (z0,zend), (x0,xend)) """

    if repr_type == "block2vec":
        # Representations
        # block2repr = load_pkl('prim_cutout_representations', prepath='/home/awiszus/Project/TOAD-GAN/input/minecraft/')
        uniques = [u for u in block2repr.keys()]
        props = [None for _ in range(len(uniques))]
        dim = len(block2repr[uniques[0]])  # all are the same size
        level = torch.zeros(
            (1, dim, coords[0][1] - coords[0][0], coords[1][1] - coords[1][0], coords[2][1] - coords[2][0]))
    else:
        uniques = []
        props = []
        level = torch.zeros((coords[0][1] - coords[0][0], coords[1][1] - coords[1][0], coords[2][1] - coords[2][0]))

    with World(input_name, input_dir, debug=debug) as wrld:
        for j in range(coords[0][0], coords[0][1]):
            for k in range(coords[1][0], coords[1][1]):
                for l in range(coords[2][0], coords[2][1]):
                    block = wrld.get_block((j, k, l))
                    # if block.get_state().id:
                    #     print("ID: ", block.get_state().id)
                    # if block.get_state().props:
                    #     print(block.get_state().name, "props:", block.get_state().props)
                    b_name = block.get_state().name
                    if repr_type == "block2vec":
                        level[0, :, j - coords[0][0], k - coords[1][0], l - coords[2][0]] = block2repr[b_name]
                        if not props[uniques.index(b_name)]:
                            props[uniques.index(b_name)] = block.get_state().props
                    else:
                        if b_name not in uniques:
                            uniques.append(b_name)
                            props.append(block.get_state().props)
                        level[j - coords[0][0], k - coords[1][0], l - coords[2][0]] = uniques.index(b_name)

    if repr_type == "block2vec":
        oh_level = level
    else:
        oh_level = torch.zeros((1, len(uniques),) + level.shape)
        for i, tok in enumerate(uniques):
            oh_level[0, i] = (level == i)

        if repr_type == "autoencoder":
            device = next(block2repr["encoder"].parameters()).device
            oh_level = block2repr["encoder"](oh_level.to(device)).detach()

    return oh_level, uniques, props


def save_level_to_world(input_dir, input_name, start_coords, bdata_level, token_list, props=None, debug=False):
    if not props:
        props = [{} for _ in range(len(token_list))]

    with World(input_name, input_dir, debug=debug) as wrld:
        # clear area with air
        # cvs = Canvas(wrld)
        # cvs.select_rectangle(start_coords, bdata_level.shape).fill(BlockState('minecraft:air', {}))
        # fill area
        for j in range(start_coords[0], start_coords[0] + bdata_level.shape[0]):
            for k in range(start_coords[1], start_coords[1] + bdata_level.shape[1]):
                for l in range(start_coords[2], start_coords[2] + bdata_level.shape[2]):
                    block = wrld.get_block((j, k, l))
                    actual_pos = (j-start_coords[0], k-start_coords[1], l-start_coords[2])
                    block.set_state(BlockState(token_list[bdata_level[actual_pos]], props[bdata_level[actual_pos]]))


def save_oh_to_wrld_directly(input_dir, input_name, start_coords, oh_level, block2repr, repr_type, token_list=None, props=None, debug=False):
    if repr_type == "autoencoder":
        oh_level = block2repr["decoder"](oh_level).detach()
    elif repr_type == "block2vec":
        token_list = list(block2repr.keys())

    if not props:
        props = [{} for _ in range(len(token_list))]

    bdata = np.zeros(oh_level.shape[2:], 'uint8')
    with World(input_name, input_dir, debug=debug) as wrld:
        # fill area
        for j in range(start_coords[0], start_coords[0] + bdata.shape[0]):
            for k in range(start_coords[1], start_coords[1] + bdata.shape[1]):
                for l in range(start_coords[2], start_coords[2] + bdata.shape[2]):
                    act_j = j-start_coords[0]
                    act_k = k-start_coords[1]
                    act_l = l-start_coords[2]
                    if repr_type == "block2vec":
                        dists = np.zeros((len(token_list),))
                        for i, rep in enumerate(token_list):
                            dists[i] = F.mse_loss(block2repr[rep], oh_level[0, :, act_j, act_k, act_l].detach().cpu()).detach()
                        bdata[act_j, act_k, act_l] = dists.argmin()
                    else:
                        bdata[act_j, act_k, act_l] = oh_level[:, :, act_j, act_k, act_l].argmax()

                    block = wrld.get_block((j, k, l))
                    block.set_state(BlockState(token_list[bdata[act_j, act_k, act_l]], props[bdata[act_j, act_k, act_l]]))

    return bdata, token_list


def clear_empty_world(worlds_folder, empty_world_name='Curr_Empty_World'):
    src = os.path.join(worlds_folder, 'Empty_World')
    dst = os.path.join(worlds_folder, empty_world_name)
    shutil.rmtree(dst)
    shutil.copytree(src, dst)

