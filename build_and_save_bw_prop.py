import os
import sys
from time import time
import struct
from multiprocessing import Pool
from itertools import combinations
from matplotlib import pyplot as plt
# import cPickle as pkl
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial import kdtree
from scipy.spatial import Delaunay
from scipy import spatial
import scipy as sp
kdtree.node = kdtree.KDTree.node
kdtree.leafnode = kdtree.KDTree.leafnode
kdtree.innernode = kdtree.KDTree.innernode

from TGMMlibraries import lineageTree

def single_cell_propagation(params):
    C, closest_cells, dist_max, W = params
    dists = np.array([np.sum((LT.pos[C]-LT.pos[n])**2)**.5 for n in closest_cells])
    if (dists<dist_max).any():
        cells_to_keep = closest_cells[np.where(dists<dist_max)]
        W_to_keep = W[np.where(dists<dist_max)]
        subset_dist = [np.mean([LT.pos[cii] for cii in LT.predecessor[ci]], axis=0) - LT.pos[ci] 
                       for ci in cells_to_keep if not LT.is_root[ci]]
        subset_W = [W_to_keep[i] for i, ci in enumerate(cells_to_keep) if not LT.is_root[ci]]
        if subset_dist != []:
            med_distance = spatial.distance.squareform(spatial.distance.pdist(subset_dist)) * subset_W
            med = subset_dist[np.argmin(np.sum(med_distance, axis=1))]
        else:
            med = np.array([0, 0, 0])
    else:
        med = np.array([0, 0, 0])
    return C, med

def build_VF_propagation_backward(LT, t_b=0, t_e=200, nb_max=20, dist_max=200, nb_proc = 8):
    if (not hasattr(LT, 'VF')) or LT.VF == None:
        LT.VF = lineageTree(None, None, None)
        starting_cells = LT.time_nodes[t_b]
        unique_id = 0
        LT.VF.time_nodes = {t_b: []}
        for C in starting_cells:
            # C_tmp = CellSS(unique_id=unique_id, id=unique_id, M=LT.VF.R, time = t_b, pos = C.pos)
            i = LT.VF.get_next_id()
            LT.VF.nodes.append(i)
            LT.VF.time_nodes[t_b].append(i)
            LT.VF.roots.append(i)
            LT.VF.pos[i]=LT.pos[C]

    from time import time

    # Hack to allow pickling of kdtrees for muLTiprocessing
    kdtree.node = kdtree.KDTree.node
    kdtree.leafnode = kdtree.KDTree.leafnode
    kdtree.innernode = kdtree.KDTree.innernode

    gg_line = '\r%03d: GG done (%.2f s); '
    prop_line = 'P done (%.2f s); '
    add_line = 'Add done (%.2f s); '
    fusion_line = 'F done (%.2f s); '
    nb_cells_line = '#C: %06d'

    full_length = len(gg_line) + 3 + 5 +\
                  len(prop_line) + 5 +\
                  len(add_line) + 5 + \
                  len(fusion_line) + 5 + \
                  len(nb_cells_line) + 6

    
    for t in range(t_b, t_e, -1):
        tic = time()
        to_check_VF = LT.VF.time_nodes[t]

        idx3d, to_check_LT = LT.get_idx3d(t)

        LT.VF.time_nodes[t-1] = []
        # tmp = np.array(to_check_LT)

        Gabriel_graph = LT.get_gabriel_graph(t)
    
        GG_pred = LT.get_gabriel_graph(t-1) if t_e < t else {}

        GG_succ = LT.get_gabriel_graph(t+1) if t < t_b else {}


        gg_time = time() - tic

        tagged = set()
        cell_mapping_LT_VF = {}
        acc = 0
        mapping = []
        for C in to_check_VF:
            C_LT = to_check_LT[idx3d.query(LT.VF.pos[C])[1]]
            cell_mapping_LT_VF.setdefault(C_LT, []).append(C)
            # mapping += [(C, idx3d, nb_max, dist_max, tmp, LT.roots, LT.pos, LT.VF.pos, LT.successor, LT.LT.predecessor)]
            if not C_LT in tagged:
                C_LT_pred = set(LT.predecessor.get(C_LT, []))
                C_LT_pred_2 = set([ci for pred in LT.predecessor.get(C_LT, []) for ci in GG_pred.get(pred, set())])
                C_LT_succ = set(LT.successor.get(C_LT, []))
                C_LT_succ_2 = set([ci for succ in LT.successor.get(C_LT, []) for ci in GG_succ.get(succ, set())])
                N = list(Gabriel_graph.get(C_LT, set()))
                weight_mat = np.ones_like(N)
                N_pred = {n: LT.predecessor.get(n, []) for n in N}
                N_succ = {n: LT.successor.get(n, []) for n in N}
                for i, n in enumerate(N):
                    for n_predii in N_pred[n]:
                        if C_LT_pred.intersection(GG_pred.get(n_predii, set())):
                            W = 4
                        elif C_LT_pred_2.intersection(GG_pred.get(n_predii, set())):
                            W = 2
                        else:
                            W = 1
                    for n_succii in N_succ[n]:
                        if C_LT_succ.intersection(GG_succ.get(n_succii, set())):
                            W += 4
                        elif C_LT_succ_2.intersection(GG_succ.get(n_succii, set())):
                            W += 2
                        else:
                            W += 0
                    weight_mat[i] = W
                mapping += [(C_LT, np.array(list(N)), dist_max, weight_mat)]
                tagged.add(C_LT)

        out = []

        if nb_proc<2:
            for params in mapping:
                out += [single_cell_propagation(params)]
        else:
            pool = Pool(processes=nb_proc)
            out = pool.map(single_cell_propagation, mapping)
            pool.terminate()
            pool.close()
            # out= out.get()
        for C_LT, med in out:
            for C_VF in cell_mapping_LT_VF[C_LT]:
                LT.VF.add_node(t-1, C_VF, LT.VF.pos[C_VF] + med, reverse = True)
            # C_next = LT.VF.get_next_id()
            # # C_next = CellSS(unique_id, unique_id, M=C, time = t-1, pos= C.pos + med)
            # LT.VF.time_nodes[t-1].append(C_next)
            # LT.VF.successor.setdefauLT(C, []).append(C_next)
            # LT.VF.edges.append((C, C_next))
            # LT.VF.nodes.append(C_next)
            # LT.VF.pos[C_next] = LT.VF.pos[C] + med

        idx3d, to_check_LT = LT.get_idx3d(t-1)
        to_check_VF = LT.VF.time_nodes[t-1]

        sys.stdout.write('\b'*(full_length) + ' '*(full_length))
        sys.stdout.flush()
        sys.stdout.write(gg_line%(t, gg_time))
        sys.stdout.flush()

        sys.stdout.write(prop_line%(time() - tic))
        sys.stdout.flush()

        if not LT.spatial_density.has_key(to_check_LT[0]):
            LT.compute_spatial_density(t-1, t-1, nb_max)

        idx3d, to_check_VF = LT.VF.get_idx3d(t-1)[:2]
        dist_to_VF, equivalence = idx3d.query([LT.pos[c] for c in to_check_LT], 1)
        tmp = np.array([dist_to_VF[i]/LT.spatial_density[c] for i, c in enumerate(to_check_LT)])
        to_add = [to_check_LT[i] for i in np.where(tmp>1.25)[0]]
        for C in to_add:
            LT.VF.add_node(t-1, None, LT.pos[C])

        sys.stdout.write(add_line%(time() - tic))
        sys.stdout.flush()
        # idx3d, to_check_LT = LT.get_idx3d(t-1)
        # to_check_VF = LT.VF.time_nodes[t-1]

        # dist_to_VF, equivalence = idx3d.query([LT.VF.pos[c] for c in to_check_VF], 1)
        # # equivalence = equivalence[:,1]

        # count = np.bincount(equivalence)
        # LT_too_close, = np.where(count > 1)
        # TMP = []
        # for C_LT in LT_too_close:
        #     to_potentially_fuse, = np.where(equivalence == C_LT)
        #     pos_tmp = [LT.VF.pos[to_check_VF[c]] for c in to_potentially_fuse]
        #     dist_tmp = spatial.distance.squareform(spatial.distance.pdist(pos_tmp))
        #     dist_tmp[dist_tmp==0] = np.inf
        #     if (dist_tmp<LT.spatial_density[to_check_LT[C_LT]]/2.).any():
        #         to_fuse = np.where(dist_tmp == np.min(dist_tmp))[0]
        #         c1, c2 = to_potentially_fuse[list(to_fuse)][:2]
        #         to_check_VF[c1], to_check_VF[c2]
        #         TMP.append([to_check_VF[c1], to_check_VF[c2]])

        # for c1, c2 in TMP:
        #     # new_pos = np.mean([LT.VF.pos[c1], LT.VF.pos[c2]], axis = 0)
        #     if c1 != c2:
        #         LT.VF.fuse_nodes(c1, c2)

        # sys.stdout.write(fusion_line%(time() - tic))
        # sys.stdout.flush()
        
        sys.stdout.write(nb_cells_line%i)
        sys.stdout.flush()
        # sys.stdout.write("\n")
        # sys.stdout.flush()

        # if len([k for k, s in LT.VF.successor.iteritems() if len(s)>1]) > 0:
        #     print 'oupsies t:', t


    LT.VF.t_b = t_e
    LT.VF.t_e = t_b

    return LT.VF

def prune_tracks(VF, mapping_LT_to_VF):
    to_keep = {}
    for v in mapping_LT_to_VF.itervalues():
        for c in v:
            to_keep[c] = True

    to_keep_tmp = {}
    for c in to_keep.iterkeys():
        m = c
        d = c
        for i in range(5):
            m = VF.successor.get(m, [m])[0]
            d = VF.predecessor.get(d, [d])[0]
            to_keep_tmp[m] = True
            to_keep_tmp[d] = True
        to_keep_tmp[c] = True

    to_keep_final = {}
    for c in to_keep_tmp.iterkeys():
        m = c
        d = c
        i = 0
        while (    to_keep_tmp.get(VF.successor.get(m, [m])[0], False)
               and to_keep_tmp.get(VF.predecessor.get(d, [d])[0], False)
               and i < 5):
            m = VF.successor.get(m, [m])[0]
            d = VF.predecessor.get(d, [d])[0]
            i += 1

        to_keep_final[c] = (i == 5)

    VF.to_take_time = {}
    for C in VF.nodes:
        if to_keep_final.get(C, False) and C in VF.time:
            VF.to_take_time.setdefault(VF.time[C], []).append(C)

def get_gabriel_graph_for_parallel(params):
    t = params
    if not hasattr(LT, 'Gabriel_graph'):
        LT.Gabriel_graph = {}

    if not LT.Gabriel_graph.has_key(t):
        idx3d, nodes = LT.get_idx3d(t)

        data_corres = {}
        data = []
        for i, C in enumerate(nodes):
            data.append(LT.pos[C])
            data_corres[i] = C

        tmp = Delaunay(data)

        delaunay_graph = {}

        for N in tmp.simplices:
            for e1, e2 in combinations(np.sort(N), 2):
                delaunay_graph.setdefault(e1, set([])).add(e2)

        Gabriel_graph = {}

        for e1, ni in delaunay_graph.iteritems():
            ni = list(ni)
            pos_e1 = data[e1]
            distances = np.array([LT._dist_v(pos_e1, data[e2]) for e2 in ni])
            sorted_neighbs = list(np.array(ni)[np.argsort(distances)])
            tmp_pt = sorted_neighbs.pop(0)
            while not tmp_pt is None:
                if len(idx3d.query_ball_point((data[tmp_pt] + pos_e1)/2., 
                                              (LT._dist_v(pos_e1, data[tmp_pt])/2)-10**-1))==0:
                    Gabriel_graph.setdefault(nodes[e1], set()).add(nodes[tmp_pt])
                    # Gabriel_graph.setdefault(nodes[tmp_pt], set()).add(nodes[e1])
                if sorted_neighbs != []:
                    tmp_pt = sorted_neighbs.pop(0)
                else:
                    tmp_pt = None
    else:
        Gabriel_graph = LT.Gabriel_graph[t]

    return t, Gabriel_graph

def parallel_gabriel_graph_preprocess(LT, nb_proc = 24):
    mapping = []
    if not hasattr(LT, 'Gabriel_graph'):
        LT.Gabriel_graph = {}
    for t in xrange(LT.t_b, LT.t_e + 1):
        if not LT.Gabriel_graph.has_key(t):
            mapping += [(t)]
    if nb_proc<2:
        out = []
        for params in mapping:
          out += [get_gabriel_graph_for_parallel(params)]
    else:
        pool = Pool(processes=nb_proc)
        out = pool.map(get_gabriel_graph_for_parallel, mapping)
        pool.terminate()
        pool.close()
    for t, G_g in out:
        LT.Gabriel_graph[t] = G_g

def write_header_am_2(f, nb_points, length):
    f.write('# AmiraMesh 3D ASCII 2.0\n')
    f.write('define VERTEX %d\n'%(nb_points*2))
    f.write('define EDGE %d\n'%nb_points)
    f.write('define POINT %d\n'%((length)*nb_points))
    f.write('Parameters {\n')
    f.write('\tContentType "HxSpatialGraph"\n')
    f.write('}\n')

    f.write('VERTEX { float[3] VertexCoordinates } @1\n')
    f.write('EDGE { int[2] EdgeConnectivity } @2\n')
    f.write('EDGE { int NumEdgePoints } @3\n')
    f.write('POINT { float[3] EdgePointCoordinates } @4\n')
    f.write('VERTEX { float Vcolor } @5\n')
    f.write('VERTEX { int Vbool } @6\n')
    f.write('EDGE { float Ecolor } @7\n')
    f.write('VERTEX { int Vbool2 } @8\n')

def write_to_am_2(path_format, LT_to_print, t_b = None, t_e = None, length = 5, manual_labels = None, 
                  default_label = 5, new_pos = None):
    if not hasattr(LT_to_print, 'to_take_time'):
        LT_to_print.to_take_time = LT_to_print.time_nodes
    if t_b is None:
        t_b = min(LT_to_print.to_take_time.keys())
    if t_e is None:
        t_e = max(LT_to_print.to_take_time.keys())
    if new_pos is None:
        new_pos = LT_to_print.pos

    if manual_labels is None:
        manual_labels = {}
    for t in range(t_b, t_e + 1):
        f = open(path_format%t, 'w')
        nb_points = len(LT_to_print.to_take_time[t])
        write_header_am_2(f, nb_points, length)
        points_v = {}
        for C in LT_to_print.to_take_time[t]:
            C_tmp = C
            positions = []
            for i in xrange(length):
                C_tmp = LT_to_print.predecessor.get(C_tmp, [C_tmp])[0]
                positions.append(new_pos[C_tmp])
            points_v[C] = positions

        f.write('@1\n')
        for i, C in enumerate(LT_to_print.to_take_time[t]):
            f.write('%f %f %f\n'%tuple(points_v[C][0]))
            f.write('%f %f %f\n'%tuple(points_v[C][-1]))

        f.write('@2\n')
        for i, C in enumerate(LT_to_print.to_take_time[t]):
            f.write('%d %d\n'%(2*i, 2*i+1))

        f.write('@3\n')
        for i, C in enumerate(LT_to_print.to_take_time[t]):
            f.write('%d\n'%(length))

        f.write('@4\n')
        tmp_velocity = {}
        for i, C in enumerate(LT_to_print.to_take_time[t]):
            for p in points_v[C]:
                f.write('%f %f %f\n'%tuple(p))

        f.write('@5\n')
        for i, C in enumerate(LT_to_print.to_take_time[t]):
            f.write('%f\n'%(manual_labels.get(C, default_label)))
            f.write('%f\n'%(0))

        f.write('@6\n')
        for i, C in enumerate(LT_to_print.to_take_time[t]):
            f.write('%d\n'%(int(manual_labels.get(C, default_label) != default_label)))
            f.write('%d\n'%(0))
        
        f.write('@7\n')
        for i, C in enumerate(LT_to_print.to_take_time[t]):
            f.write('%f\n'%(np.linalg.norm(points_v[C][0] - points_v[C][-1])))

        f.write('@8\n')
        for i, C in enumerate(LT_to_print.to_take_time[t]):
            f.write('%d\n'%(1))
            f.write('%d\n'%(0))

        f.close()

def write_header_am(f, nb_points, length):
    f.write('# AmiraMesh 3D ASCII 2.0\n')
    f.write('define VERTEX %d\n'%nb_points)
    f.write('define EDGE %d\n'%nb_points)
    f.write('define POINT %d\n'%(3*nb_points))
    f.write('Parameters {\n')
    f.write('\tContentType "HxSpatialGraph"\n')
    f.write('}\n')

    f.write('VERTEX { float[3] VertexCoordinates } @1\n')
    f.write('EDGE { int[2] EdgeConnectivity } @2\n')
    f.write('EDGE { int NumEdgePoints } @3\n')
    f.write('POINT { float[3] EdgePointCoordinates } @4\n')
    f.write('VERTEX { float Vcolor } @5\n')
    f.write('VERTEX { int Vbool } @6\n')
    f.write('EDGE { float Ecolor } @7\n')

def write_to_am(path_format, LT, t_b = None, t_e = None, length = 5, manual_labels = None):
    if t_b is None:
        t_b = min(LT.to_take_time.keys())
    if t_e is None:
        t_e = max(LT.to_take_time.keys())

    if manual_labels is None:
        manual_labels = {}
    for t in range(t_b, t_e + 1):
        f = open(path_format%t, 'w')
        nb_points = len(LT.to_take_time[t])
        write_header_am(f, nb_points, length)
        if LT.to_take_time[t] != []:
            f.write('@1\n')
            for i, C in enumerate(LT.to_take_time[t]):
                f.write('%f %f %f\n'%tuple(LT.pos[C]))

            f.write('@2\n')
            for i, C in enumerate(LT.to_take_time[t]):
                f.write('%d %d\n'%(i, i))

            f.write('@3\n')
            for i, C in enumerate(LT.to_take_time[t]):
                f.write('%d\n'%3)#(length+1))

            f.write('@4\n')
            tmp_velocity = {}
            for i, C in enumerate(LT.to_take_time[t]):
                C_tmp = C
                positions = []
                for i in xrange(length*2):
                    positions.append(LT.pos[C_tmp])
                    C_tmp = LT.predecessor.get(C_tmp, [C_tmp])[0]
                final_pos = np.mean(positions[length-1:], axis = 0)
                f.write('%f %f %f\n'%tuple(LT.pos[C]))
                f.write('%f %f %f\n'%tuple(final_pos))
                f.write('%f %f %f\n'%tuple(LT.pos[C]))
                tmp_velocity[C] = ((np.sum((final_pos - LT.pos[C])**2))**.5)
                # for pi in p[::-1]:
                #     f.write('%f %f %f\n'%tuple(p))

            f.write('@5\n')
            for i, C in enumerate(LT.to_take_time[t]):
                f.write('%f\n'%(manual_labels.get(C, 5)))

            f.write('@6\n')
            for i, C in enumerate(LT.to_take_time[t]):
                f.write('%d\n'%(int(manual_labels.get(C, 5) != 5)))
            
            f.write('@7\n')
            for i, C in enumerate(LT.to_take_time[t]):
                f.write('%f\n'%(tmp_velocity[C]))
        f.close()

def mapping_VF_to_LT(params):
    t, = params
    cells_VF = VF.time_nodes[t]
    cells_LT = LT.time_nodes[t]
    idx3d, mapping_VF = VF.get_idx3d(t)

    positions_LT = [LT.pos[C] for C in cells_LT]
    mapping = idx3d.query(positions_LT, 10)[1]
    final_mapping = {}
    for i, C_neighs in enumerate(mapping):
        final_mapping[cells_LT[i]] = [mapping_VF[ci] for ci in C_neighs]
    return final_mapping

def parallel_mapping(tb, te, nb_proc = -1):
    mapping = []
    for t in range(tb, te+1):
        mapping += [[t]]

    if nb_proc == 1:
        out = []
        for params in mapping:
            out += [mapping_VF_to_LT(params)]
    else:
        pool = Pool(processes = nb_proc)
        out = pool.map(mapping_VF_to_LT, mapping)
        pool.terminate()
        pool.close()

    mapping_VF_to_LT_out = {}
    for tmp in out:
        mapping_VF_to_LT_out.update(tmp)

    return mapping_VF_to_LT_out

def GG_to_bin(gg, fname):

    nodes_list = []
    time_list = []
    for t, gg_t in gg.iteritems():
        time_list += [t]
        len_ = 0
        for node, neighbors in gg_t.iteritems():
            to_add = [-(node + 1)] + [neighb + 1 for neighb in neighbors]
            nodes_list += to_add
            len_ += len(to_add)
        time_list += [len_]

    f = open(fname, 'wb')
    f.write(struct.pack('q', len(time_list)))
    f.write(struct.pack('q', len(nodes_list)))
    f.write(struct.pack('q'*len(time_list), *time_list))
    f.write(struct.pack('q'*len(nodes_list), *nodes_list))
    f.close()


def GG_from_bin(fname):
    q_size = struct.calcsize('q')
    f = open(fname)
    len_time_list = struct.unpack('q', f.read(q_size))[0]
    len_nodes_list = struct.unpack('q', f.read(q_size))[0]
    time_list = list(np.reshape(struct.unpack('q'*len_time_list, 
                                              f.read(q_size*len_time_list)),
                                (len_time_list/2, 2)))
    nodes_list = list(struct.unpack('q'*len_nodes_list, f.read(q_size*len_nodes_list)))
    f.close()

    gg ={}
    pos = 0
    for t, len_ in time_list:
        gg[t] = {}
        for n in nodes_list[pos:pos + len_]:
            if n < 0:
                current_node = -n - 1
                gg[t][current_node] = set()
            else:
                gg[t][current_node].add(n - 1)
        pos += len_

    return gg

def get_tracks_length(LT):
    mothers = [k for k, take in LT.is_root.iteritems() if take]
    tracks_length = []
    progenies_length = []
    for m in mothers:
        t_start = LT.time[m]
        t_end = LT.time[m]
        to_treat = [m]
        while to_treat != []:
            curr_cell = to_treat.pop()
            track_length = 1
            while len(LT.successor.get(curr_cell, [])) == 1:
                curr_cell = LT.successor[curr_cell][0]
                t_end = max(t_end, LT.time[curr_cell])
                track_length += 1
            tracks_length += [track_length]
            to_treat += LT.successor.get(curr_cell, [])
        progenies_length += [t_end - t_start + 1]
    return tracks_length, progenies_length

def write_stats(LT, path):
    nb_divs = len([m for m, ds in LT.successor.iteritems() if len(ds) > 1])
    nb_cells = len(LT.nodes)
    nb_cell_creation = np.sum(LT.is_root.values())
    tracks_length, progenies_length = get_tracks_length(LT)

    f = open(path + 'out_data.txt', 'w')
    f.write(('%d divs\n%d cells\n%d cell creations\n' +\
                'average track length: %.2f\n' + \
                'average progeny length: %.2f\n' +\
                '#tracks smaller than 5: %d\n' +\
                '#progeny smaller than 5: %d\n')%(nb_divs, nb_cells, nb_cell_creation, 
                                             np.mean(tracks_length), np.mean(progenies_length),
                                             np.sum(np.array(tracks_length)<=5),
                                             np.sum(np.array(progenies_length)<=5)))
    f.close()
    fig = plt.figure(figsize = (10, 8))
    ax = fig.add_subplot(111)
    ax.hist(tracks_length, bins = 50)
    ax.set_title('Progenies length distribution')
    plt.savefig(path + 'track_length_distribution.pdf')
    plt.clf()
    fig = plt.figure(figsize = (10, 8))
    ax = fig.add_subplot(111)
    ax.hist(progenies_length, bins = 50)
    ax.set_title('Track length distribution')
    plt.savefig(path + 'progenies_length_distribution.pdf')
    plt.clf()

def add_manual_tracking(path_to_manual, v, VF, drift_correction):
    manual_CM = lineageTree(path_to_manual, MaMuT = True)
    # VF_copy = deepcopy(VF)
    corres_M_VF = {}
    manual_track = {}

    for t in range(manual_CM.t_b, min(520, manual_CM.t_e+1)):
        for c in manual_CM.time_nodes.get(t, []):
            if c in manual_CM.successor or c in manual_CM.predecessor:
                id_ = VF.get_next_id()
                corres_M_VF[c] = id_
                VF.time_nodes.setdefault(t, []).append(id_)
                VF.nodes.append(id_)
                VF.pos[id_] = manual_CM.pos[c] - drift_correction[t]
                manual_track[id_] = v
                VF.time[id_] = t

    for e in manual_CM.edges:
        if corres_M_VF.has_key(e[1]):
            if np.abs(VF.time[corres_M_VF[e[0]]] - VF.time[corres_M_VF[e[1]]]) != 1:
                print e, VF.time[corres_M_VF[e[0]]], VF.time[corres_M_VF[e[1]]]
            VF.successor.setdefault(corres_M_VF[e[0]], []).append(corres_M_VF[e[1]])
            VF.predecessor.setdefault(corres_M_VF[e[1]], []).append(corres_M_VF[e[0]])

    return manual_track, corres_M_VF

def build_equivalent(VF, manual_track):
    used = {}
    tracks = []
    for c, l in manual_track.iteritems():
        if not used.get(c, False):
            used[c] = True
            track = {VF.time[c]: [c]}
            to_treat = [c]
            while to_treat!=[]:
                tmp = to_treat.pop()
                next_ = VF.successor.get(tmp, [])
                for ci in next_:
                    if not used.get(ci, False):
                        track.setdefault(VF.time[ci], []).append(ci)
                        used[ci] = True
                        to_treat.append(ci)

                if VF.predecessor.has_key(tmp):
                    next_ = VF.predecessor[tmp]
                    for ci in next_:
                        if not used.get(ci, False):
                            track.setdefault(VF.time[ci], []).append(ci)
                            used[ci] = True
                            to_treat.append(ci)

            tracks.append(track)

    return tracks


def compute_track_distance(VF, track, c, m_pos, times, starting_time, LT_tracks):
    used = {}
    t = starting_time
    track_VF = {t: [c]}
    to_treat = [c]
    while to_treat!=[]:
        tmp = to_treat.pop()
        for ci in VF.successor.get(tmp, []):
            if not used.get(ci, False):
                track_VF.setdefault(VF.time.get(ci, VF.t_e), []).append(ci)
                used[ci] = True
                to_treat.append(ci)

    t = starting_time
    to_treat = list(VF.predecessor.get(c, []))
    while to_treat!=[]:
        tmp = to_treat.pop()
        for ci in VF.predecessor.get(tmp, []):
            if not used.get(ci, False):
                track_VF.setdefault(VF.time.get(ci, VF.t_e), []).append(ci)
                used[ci] = True
                to_treat.append(ci)

    track_VF = {t: track_VF.get(t, []) for i, t in enumerate(times)}
    m_pos_VF = np.array([np.mean([VF.pos[c] for c in track_VF[t]], axis = 0) for t in times if track_VF[t] != []])
    final_track_VF = np.array(sum([track_VF[t] for t in times if track_VF[t] != []], []))
    m_pos = np.array([np.mean([LT_tracks.pos[c] for c in track[t]], axis = 0) for t in times if track_VF[t] != []])
    final_time = [t for t in times if track_VF[t] != []]

    return np.sqrt(np.sum((m_pos_VF - m_pos)**2, axis = 1))[1:], final_time, final_track_VF# / (np.sqrt(np.sum((m_pos[0] - m_pos)**2, axis = 1)))[1:]

def get_all_track_distances(VF, tracks, LT_tracks):
    distances = {}
    best_match = {}
    common_times = {}
    final_tracks_VF = []
    for track in tracks:
        final_times = []
        tracks_VF = []
        times = sorted(track.keys())
        m_pos = np.array([np.mean([LT_tracks.pos[ci] for ci in track[t]], axis = 0) for t in times])
        tmp = []
        for i, pi in enumerate(m_pos):
            idx3d, VF_equ = VF.get_idx3d(times[i])
            c = VF_equ[idx3d.query(pi)[1]]
            dist, time, track_VF = compute_track_distance(VF, track, c, m_pos, times, times[i], LT_tracks)
            tmp.append(dist)
            tracks_VF.append(track_VF)
            final_times.append(time)
        ids = []
        for di in tmp:
            if len(di)/float(len(track))>0.75:
                ids.append(np.mean(di))
            else:
                ids.append(np.inf)
        if ids != []:
            distances[track[np.min(track.keys())][0]] = tmp[np.argmin(ids)]
            best_match[track[np.min(track.keys())][0]] = times[np.argmin(ids)]
            common_times[track[np.min(track.keys())][0]] = final_times[np.argmin(ids)]
            final_tracks_VF.append(tracks_VF[np.argmin(ids)])

    return distances, best_match, common_times, final_tracks_VF



def get_length(LT, c, max_val = 1e4):
    length = 0
    c_tmp = [c]
    while (len(LT.successor.get(c_tmp[-1], [])) == 1 or 
           not LT.to_keep.get(LT.successor.get(c_tmp[-1], [-1])[0], True) or 
           not LT.to_keep.get(LT.successor.get(c_tmp[-1], [-1, -1])[1], True)):
        if LT.to_keep.get(LT.successor[c_tmp[-1]][0], True):
            c_tmp += [LT.successor[c_tmp[-1]][0]]
        else:
            c_tmp += [LT.successor[c_tmp[-1]][1]]
    if c_tmp[-1] in LT.successor:
        return max_val, None
    else:
        return len(c_tmp), c_tmp

def remove_cells(LT, threshold = 2):
    new_born_cells = []
    for mother, sisters in LT.successor.iteritems():
        if len(sisters) > 1:
            nb_to_keep = np.sum([LT.to_keep.get(sis, True) for sis in sisters])
            if nb_to_keep > 1:
                new_born_cells += sisters

    new_born_cells = set(new_born_cells).union(set([r for r in LT.roots if LT.to_keep.get(r, True)]))
    to_remove = {}
    for c in new_born_cells:
        len_, p = get_length(LT, c)
        if len_ <= threshold:
            # if not LT.predecessor.get(c, -1) in mother_removed:
            to_remove.setdefault(LT.predecessor.get(c, [-1])[0], []).append((p, len_))

    nb_removed = 0
    # LT.to_keep = {}
    for branch, len_ in to_remove.pop(-1, []):
        nb_removed += 1
        for c in branch:
            LT.to_keep[c] = False

    for m, branches in to_remove.iteritems():
        nb_removed += 1
        if len(branches) > 1:
            if branches[0][1] > branches[1][1]:
                for c in branches[1][0]:
                    LT.to_keep[c] = False
            else:
                for c in branches[0][0]:
                    LT.to_keep[c] = False
        else:
            for c in branches[0][0]:
                LT.to_keep[c] = False

    return nb_removed

def get_short_live_daughters(LT, threshold = 2, rounds_max = np.inf):
    LT.to_keep = {}
    rounds = 0
    while remove_cells(LT, threshold = threshold) > 0 and rounds < rounds_max:
        rounds += 1
    return rounds


def write_to_prune(LT, file_format_input, file_format_output):
    old_id_to_new = {}
    old_lin_to_new = {}
    lin_id = 0
    removed = []
    for t in range(LT.t_b, LT.t_e+1):
        new_id = 0
        print t,
        if t%10==0:
            print
        tree = ET.parse(file_format_input%t)
        root = tree.getroot()
        to_remove = []
        nb = 0
        for it in root:
            try:
                M_id, pos, cell_id, old_lin_id = (int(it.attrib['parent']), 
                                               [float(v) for v in it.attrib['m'].split(' ') if v!=''], 
                                               int(it.attrib['id']), int(it.attrib['lineage']))
                # if not (t, cell_id) in LT.time_id or not LT.to_keep.get(LT.time_id[(t, cell_id)], True):
                if not LT.to_keep.get(LT.time_id.get((t, cell_id), -1), True):
                    to_remove += [it]
                    removed += [LT.time_id.get((t, cell_id), -1)]
                    nb += 1
                else:

                    # if t == 201 and old_lin_to_new.get(old_lin_id, 0) == 0:#old_id_to_new.get((t-1, M_id), 0) == 12:
                        # print 'YO'
                        # break
                    # break
                    it.set('parent', str(old_id_to_new.get((t-1, M_id), -1)))
                    it.set('id', str(new_id))
                    old_id_to_new[(t, cell_id)] = new_id
                    new_id += 1
                    # if not old_lin_to_new.has_key(old_lin_id):
                    #     old_lin_to_new[old_lin_id] = lin_id
                    #     lin_id += 1
                    # it.set('lineage', str(old_lin_to_new[old_lin_id]))

            except Exception as e:
                to_remove += [it]
                nb += 1
                
        [root.remove(it) for it in to_remove]
        tree.write(file_format_output%t)

    return removed


def read_param_file():
    p_param = raw_input('Please enter the path to the parameter file/folder:\n')
    p_param = p_param.replace('"', '')
    p_param = p_param.replace("'", '')
    p_param = p_param.replace(" ", '')
    if p_param[-4:] == '.csv':
        f_names = [p_param]
    else:
        f_names = [os.path.join(p_param, f) for f in os.listdir(p_param) if '.csv' in f and not '~' in f]
    for file_name in f_names:
        f = open(file_name)
        lines = f.readlines()
        f.close()
        param_dict = {}
        i = 0
        nb_lines = len(lines)
        while i < nb_lines:
            l = lines[i]
            split_line = l.split(',')
            param_name = split_line[0]
            if param_name in ['labels', 'downsampling']:
                name = param_name
                out = []
                while (name == param_name or param_name == '') and  i < nb_lines:
                    out += [int(split_line[1])]
                    i += 1
                    if i < nb_lines:
                        l = lines[i]
                        split_line = l.split(',')
                        param_name = split_line[0]
                param_dict[name] = np.array(out)
            else:
                param_dict[param_name] = split_line[1].strip()
                i += 1
            if param_name == 'time':
                param_dict[param_name] = int(split_line[1])
        path_LT = param_dict.get('path_to_LT', '.')
        path_VF = param_dict.get('path_to_VF', '.')
        path_mask = param_dict.get('path_to_mask', '.')
        t = param_dict.get('time', 0)
        path_out_am = param_dict.get('path_to_am', '.')
        labels = param_dict.get('labels', [])
        DS = param_dict.get('downsampling', [])

    return (path_LT, path_VF, path_mask, t, path_out_am, labels, DS)

datas = ['/media/X/SV1/14-05-21/ParameterSweep2/TGMM/140521/Tau15Thr100/FullTS/GMEMtracking3D_2016_6_10_14_41_45/XML_finalResult_lht/',
         '/media/X/SV1/14-05-21/ParameterSweep2/TGMM_CV2/SelectParameterScreen/FinalScreen/GMEMtracking3D_ECC9/XML_finalResult_lht/',
         '/media/X/SV1/14-05-21/ParameterSweep2/TGMM_CV2/SelectParameterScreen/FinalScreen/ECC9_CV2/GMEMtracking3D_ECC9_CV2/XML_finalResult_lht/',
         '/media/X/SV1/14-05-21/ParameterSweep2/TGMM_CV2/SelectParameterScreen/FinalScreen/ECC9_CV3.1/GMEMtracking3D_2016_12_1_17_46_22/XML_finalResult_lht/',
         '/media/X/SV1/14-05-21/ParameterSweep2/TGMM_CV2/SelectParameterScreen/FinalScreen/ECC9_CV3.2/GMEMtracking3D_2016_12_2_19_43_39/XML_finalResult_lht/',
         '/media/X/SV1/14-05-21/ParameterSweep2/TGMM_CV2/SelectParameterScreen/FinalScreen/ECC9_NoDiv/GMEMtracking3D_2016_12_2_23_4_51/XML_finalResult_lht/',
         '/media/X/SV1/KM_14-08-13/TGMM/GMEMtracking3D_ECC9_CV3.1/XML_finalResult_lht/',
         '/media/X/SV1/KM_14-10-09/TGMM/GMEMtracking3D_ECC9_CV3.1/XML_finalResult_lht/',
         '/media/X/SV1/KM_15-04-03/TGMM/GMEMtracking3D_ECC9_CV3.1/XML_finalResult_lht/',
         '/media/R/SV1/KM_15-11-04/TGMM/GMEMtracking3D_ECC9_CV3.1/XML_finalResult_lht/',
         '/media/R/SV1/KM_15-11-14/TGMM/GMEMtracking3D_ECC9_CV3.1/XML_finalResult_lht/',
         '/media/R/SV1/KM_15-11-17/TGMM/GMEMtracking3D_ECC9_CV3.1/XML_finalResult_lht/',
         '/media/R/SV1/KM_15-12-05/TGMM/GMEMtracking3D_ECC9_CV3.1/XML_finalResult_lht/',
         '/media/R/SV1/KM_16-01-12/TGMM/GMEMtracking3D_ECC9_CV3.1/XML_finalResult_lht/',
         '/media/X/SV1/KM_16-03-15/TGMM/GMEMtracking3D_ECC9_CV3.1/XML_finalResult_lht/',
         '/media/X/SV1/KM_16-03-26/TGMM/GMEMtracking3D_ECC9_CV3.1/XML_finalResult_lht/',
         '/media/R/SV1/KM_16-03-28/TGMM/GMEMtracking3D_ECC9_CV3.1/XML_finalResult_lht/',
         '/media/R/SV1/KM_16-06-01/TGMM/GMEMtracking3D_ECC9_CV3.1/XML_finalResult_lht/',
         '/media/R/SV1/KM_16-06-16/TGMM/GMEMtracking3D_ECC9_CV3.1/XML_finalResult_lht/',
         '/media/R/SV1/KM_16-07-23/TGMM/GMEMtracking3D_ECC9_CV3.1/XML_finalResult_lht/',
         '/media/R/SV1/KM_16-07-26/TGMM/GMEMtracking3D_ECC9_CV3.1/XML_finalResult_lht/',
         '/media/R/SV1/KM_16-07-29/TGMM/GMEMtracking3D_ECC9_CV3.1/XML_finalResult_lht/',
         '/media/R/SV1/KM_16-10-13/TGMM/GMEMtracking3D_ECC9_CV3.1/XML_finalResult_lht/',
         '/media/R/SV1/KM_17-02-11/TGMM/GMEMtracking3D_2017_4_17_22_9_57/XML_finalResult_lht']

base_path_out = '/data/Mouse-Project/Binary-trees/'
list_path_out = [base_path_out + '14-05-21/CDroso/',
                 base_path_out + '14-05-21/ECC9_no_C/',
                 base_path_out + '14-05-21/ECC9_CV2/',
                 base_path_out + '14-05-21/ECC9_CV3.1/',
                 base_path_out + '14-05-21/ECC9_CV3.2/',
                 base_path_out + '14-05-21/ECC9_no_D/',
                 base_path_out + 'KM_14-08-13/',
                 # base_path_out + 'KM_15-11-14/',
                 # base_path_out + 'KM_16-06-16/',
                 # base_path_out + 'KM_14-08-13/',
                 base_path_out + 'KM_14-10-09/',
                 base_path_out + 'KM_15-04-03/',
                 base_path_out + 'KM_15-11-04/',
                 base_path_out + 'KM_15-11-14/',
                 base_path_out + 'KM_15-11-17/',
                 base_path_out + 'KM_15-12-05/',
                 base_path_out + 'KM_16-01-12/',
                 base_path_out + 'KM_16-03-15/',
                 base_path_out + 'KM_16-03-26/',
                 base_path_out + 'KM_16-03-28/',
                 base_path_out + 'KM_16-06-01/',
                 base_path_out + 'KM_16-06-16/',
                 base_path_out + 'KM_16-07-23/',
                 base_path_out + 'KM_16-07-26/',
                 base_path_out + 'KM_16-07-29/',
                 base_path_out + 'KM_16-10-13/',
                 base_path_out + 'KM_17-02-11/']

list_anisotropy = [5.] * len(datas)

for path_to_xml, path_out, anisotropy in zip(datas, list_path_out, list_anisotropy)[-1:]:
    files = [f for f in os.listdir(path_to_xml) if '.xml' in f]
    pos_time = len(os.path.commonprefix(files))
    times = [int(file.split('.')[0][pos_time:]) for file in files]
    tb = min(times) + 10
    te = max(times) - 10

    if not os.path.exists(path_out):
        os.makedirs(path_out)
    if not os.path.exists(path_out + 'Amira_SVF/'):
        os.makedirs(path_out + 'Amira_SVF/')
    if not os.path.exists(path_out + 'Amira_TGMM/'):
        os.makedirs(path_out + 'Amira_TGMM/')

    if not os.path.exists(path_out + 'TGMM.bin'):
        LT_main = lineageTree(file_format = path_to_xml + '/GMEMfinalResult_frame%04d.xml',
                         tb = tb, te = te, z_mult = anisotropy)
        LT_main.to_binary(path_out + 'TGMM.bin')
    else:
        LT_main = lineageTree(file_format = path_out + 'TGMM.bin')

    if not os.path.exists(path_out + 'GG.bin'):
        LT = LT_main
        global LT
        tic = time()
        parallel_gabriel_graph_preprocess(LT_main, nb_proc = 24)
        GG_to_bin(LT_main.Gabriel_graph, path_out + 'GG.bin')
        print 'Gabriel graph pre-processing:',  time() - tic
    else:
        LT_main.Gabriel_graph = GG_from_bin(path_out + 'GG.bin')

    if Tnot os.path.exists(path_out + 'VF.bin'):
        tic = time()
        VF = build_VF_propagation_backward(LT_main, t_b = te, t_e = tb, nb_max = 10, nb_proc=24)
        print 'parallel processing:',  time() - tic
        VF.to_binary(path_out + 'SVF.bin')
    else:
        VF = lineageTree(path_out + 'SVF.bin'%th)


    mapping_LT_to_VF = parallel_mapping(tb, te, 24)
    prune_tracks(VF, mapping_LT_to_VF)


    done = set()
    corresponding_track = {}
    smoothed_pos = {}
    num_track = 0
    for C in VF.nodes:
        if not C in done:
            track = [C]
            while track[-1] in VF.successor:
                track.append(VF.successor[track[-1]][0])
            while track[0] in VF.predecessor:
                track.insert(0, VF.predecessor[track[0]][0])
            pos_track = np.array([VF.pos[Ci] for Ci in track])
            X = sp.ndimage.filters.gaussian_filter1d(pos_track[:, 0], sigma = 5)
            Y = sp.ndimage.filters.gaussian_filter1d(pos_track[:, 1], sigma = 5)
            Z = sp.ndimage.filters.gaussian_filter1d(pos_track[:, 2], sigma = 5)
            track_smoothed = np.zeros_like(pos_track)
            track_smoothed[:, 0] = X
            track_smoothed[:, 1] = Y
            track_smoothed[:, 2] = Z
            smoothed_pos.update(zip(track, list(track_smoothed)))
            done.update(set(track))

    write_to_am_2(path_out + 'Amira_SVF/seg_t%04d.am', VF, t_b = None, t_e = None,
                  manual_labels = {}, default_label = 1, 
                  length = 7, new_pos = smoothed_pos)


    done = set()
    corresponding_track = {}
    smoothed_pos = {}
    num_track = 0
    all_tracks = []
    for C in LT.nodes:
        if not C in done:
            track = [C]
            while track[-1] in LT.successor:
                track.append(LT.successor[track[-1]][0])
            while track[0] in LT.predecessor:
                track.insert(0, LT.predecessor[track[0]][0])
            all_tracks += [track]
            done.update(set(track))
            pos_track = np.array([LT.pos[Ci] for Ci in track])
            X = sp.ndimage.filters.gaussian_filter1d(pos_track[:, 0], sigma = 5)
            Y = sp.ndimage.filters.gaussian_filter1d(pos_track[:, 1], sigma = 5)
            Z = sp.ndimage.filters.gaussian_filter1d(pos_track[:, 2], sigma = 5)
            track_smoothed = np.zeros_like(pos_track)
            track_smoothed[:, 0] = X
            track_smoothed[:, 1] = Y
            track_smoothed[:, 2] = Z
            smoothed_pos.update(zip(track, list(track_smoothed)))

    write_to_am_2(path_out + 'Amira_TGMM/seg_t%04d.am', LT, t_b = None, t_e = None,
                  manual_labels = {}, default_label = 1, 
                  length = 7, new_pos = smoothed_pos)


    path_to_manual = '/data/Mouse-Project/Data-sets/14-05-21/Manual-tracking/'

    drift_correction = dict(np.loadtxt(path_to_manual + 'shifts.txt'))
    drift_correction = {k: np.array([v, 0, 0]) for k, v in drift_correction.iteritems()}

    corres_M_VF = {}
    manual_track = {}
    LT_tracks = lineageTree(None)

    manual_track_tmp, corres_M_VF_tmp = add_manual_tracking(path_to_manual + '140521_DorsalAorta.xml', 1,
                                                            LT_tracks, drift_correction)
    corres_M_VF.update(corres_M_VF_tmp)
    manual_track.update(manual_track_tmp)

    manual_track_tmp, corres_M_VF_tmp = add_manual_tracking(path_to_manual + '140521_Notochord.xml', 2,
                                                            LT_tracks, drift_correction)
    corres_M_VF.update(corres_M_VF_tmp)
    manual_track.update(manual_track_tmp)

    manual_track_tmp, corres_M_VF_tmp = add_manual_tracking(path_to_manual + '140521_VE_Extended.xml', 3,
                                                            LT_tracks, drift_correction)
    corres_M_VF.update(corres_M_VF_tmp)
    manual_track.update(manual_track_tmp)

    tracks = build_equivalent(LT_tracks, manual_track)

    distances, best_match, common_times, tracks_VF = get_all_track_distances(VF, tracks, LT_tracks)

    maxs = {}
    means = {}
    for c, v in distances.iteritems():
        if list(v) != []:
            maxs.setdefault(manual_track.get(c, 0), []).append(np.max(v))
            means.setdefault(manual_track.get(c, 0), []).append(np.mean(v))

    best_match_tissue = {}
    for c, v in best_match.iteritems():
        best_match_tissue.setdefault(manual_track.get(c, 0), []).append(v)


    maxs_avg = {k:np.mean(v)*.4 for k, v in maxs.iteritems()}
    means_avg = {k:np.mean(v)*.4 for k, v in means.iteritems()}

    np.savetxt(path_out + 'stats_fw_prop_T.txt', maxs_avg.items() + means_avg.items(), fmt = '%.02f')
    import cPickle as pkl
    f = open(path_out + 'stats_fw_prop_T.pkl', 'w')
    pkl.dump(distances, f)
    f.close()

    mapping_LT_to_VF = parallel_mapping(tb, te, 24)
    prune_tracks(VF, mapping_LT_to_VF)


    done = set()
    corresponding_track = {}
    smoothed_pos = {}
    num_track = 0
    for C in VF.nodes:
        if not C in done:
            track = [C]
            while track[-1] in VF.successor:
                track.append(VF.successor[track[-1]][0])
            while track[0] in VF.predecessor:
                track.insert(0, VF.predecessor[track[0]][0])
            pos_track = np.array([VF.pos[Ci] for Ci in track])
            X = sp.ndimage.filters.gaussian_filter1d(pos_track[:, 0], sigma = 5)
            Y = sp.ndimage.filters.gaussian_filter1d(pos_track[:, 1], sigma = 5)
            Z = sp.ndimage.filters.gaussian_filter1d(pos_track[:, 2], sigma = 5)
            track_smoothed = np.zeros_like(pos_track)
            track_smoothed[:, 0] = X
            track_smoothed[:, 1] = Y
            track_smoothed[:, 2] = Z
            smoothed_pos.update(zip(track, list(track_smoothed)))
            done.update(set(track))

    if not os.path.exists(path_out + 'T_amira_t_prop/'):
        os.makedirs(path_out + 'T_amira_t_prop/')
    write_to_am_2(path_out + 'T_amira_t_prop/seg_t%04d.am', VF, t_b = None, t_e = None,
                  manual_labels = {}, default_label = 1, 
                  length = 7, new_pos = smoothed_pos)

    rounds_max = [np.inf,]

    thresholds = [1, 2, 3, 5]

    for r_m in rounds_max:
        for th in thresholds:
            if not os.path.exists(path_out + '/th_tree_%d'%th):
                os.makedirs(path_out + '/th_tree_%d'%th)
            get_short_live_daughters(LT_main, threshold = th, rounds_max = r_m)
            write_to_prune(LT_main, path_to_xml + '/GMEMfinalResult_frame%04d.xml', path_out + '/th_tree_%d'%th + '/GMEMfinalResult_frame%04d.xml')

            LT = lineageTree(file_format = path_out + '/th_tree_%d'%th + '/GMEMfinalResult_frame%04d.xml',
                                     tb = tb, te = te, z_mult = anisotropy)
            # write_stats(LT, path_out)
            if not os.path.exists(path_out + 'GG_%d.bin'):
                tic = time()
                parallel_gabriel_graph_preprocess(LT, nb_proc = 24)
                GG_to_bin(LT.Gabriel_graph, path_out + 'GG_%d.bin'%th)
                print 'Gabriel graph pre-processing:',  time() - tic
            else:
                LT.Gabriel_graph = GG_from_bin(path_out + 'GG_%d.bin'%th)

            if not os.path.exists(path_out + 'VF_%d.bin'):
                tic = time()
                VF = build_VF_propagation_backward(LT, t_b = te, t_e = tb, nb_max = 10, nb_proc=24)
                print 'parallel processing:',  time() - tic
                VF.to_binary(path_out + 'VF_%d.bin'%th)
            else:
                VF = lineageTree(path_out + 'VF_%d.bin'%th)

            # mapping_LT_to_VF = parallel_mapping(tb, te, 24)

            # prune_tracks(VF, mapping_LT_to_VF)

            # write_to_am(path_out + 'Amira_output/seg_t%04d.am', VF, t_b= None, t_e= None)


            path_to_manual = '/data/Mouse-Project/Data-sets/14-05-21/Manual-tracking/'

            drift_correction = dict(np.loadtxt(path_to_manual + 'shifts.txt'))
            drift_correction = {k: np.array([v, 0, 0]) for k, v in drift_correction.iteritems()}

            corres_M_VF = {}
            manual_track = {}
            LT_tracks = lineageTree(None)

            manual_track_tmp, corres_M_VF_tmp = add_manual_tracking(path_to_manual + '140521_DorsalAorta.xml', 1,
                                                                    LT_tracks, drift_correction)
            corres_M_VF.update(corres_M_VF_tmp)
            manual_track.update(manual_track_tmp)

            manual_track_tmp, corres_M_VF_tmp = add_manual_tracking(path_to_manual + '140521_Notochord.xml', 2,
                                                                    LT_tracks, drift_correction)
            corres_M_VF.update(corres_M_VF_tmp)
            manual_track.update(manual_track_tmp)

            manual_track_tmp, corres_M_VF_tmp = add_manual_tracking(path_to_manual + '140521_VE_Extended.xml', 3,
                                                                    LT_tracks, drift_correction)
            corres_M_VF.update(corres_M_VF_tmp)
            manual_track.update(manual_track_tmp)

            tracks = build_equivalent(LT_tracks, manual_track)

            distances, best_match, common_times, tracks_VF = get_all_track_distances(VF, tracks, LT_tracks)

            maxs = {}
            means = {}
            for c, v in distances.iteritems():
                if list(v) != []:
                    maxs.setdefault(manual_track.get(c, 0), []).append(np.max(v))
                    means.setdefault(manual_track.get(c, 0), []).append(np.mean(v))

            best_match_tissue = {}
            for c, v in best_match.iteritems():
                best_match_tissue.setdefault(manual_track.get(c, 0), []).append(v)


            maxs_avg = {k:np.mean(v)*.4 for k, v in maxs.iteritems()}
            means_avg = {k:np.mean(v)*.4 for k, v in means.iteritems()}

            np.savetxt(path_out + 'stats_fw_prop_%d.txt'%th, maxs_avg.items() + means_avg.items(), fmt = '%.02f')
            import cPickle as pkl
            f = open(path_out + 'stats_fw_prop_%d.pkl'%th, 'w')
            pkl.dump(distances, f)
            f.close()
