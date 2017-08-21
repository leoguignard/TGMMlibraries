from scipy.spatial import kdtree
import os
import xml.etree.ElementTree as ET
from copy import copy
from scipy import spatial
import numpy as np
from scipy import ndimage as nd
from matplotlib import pyplot as plt
import networkx as nx
from multiprocessing import Pool
from scipy.spatial import Delaunay
from itertools import combinations
import struct

def single_cell_propagation_BU(params):
    kdtree.node = kdtree.KDTree.node
    kdtree.leafnode = kdtree.KDTree.leafnode
    kdtree.innernode = kdtree.KDTree.innernode
    C, idx3d, nb_max, dist_max, to_check_self, R, pos, posVF, successor, predecessor = params
    dists, closest_cells = idx3d.query(posVF[C], nb_max)
    if (dists<dist_max).any():
        closest_cells = np.array(to_check_self)[list(closest_cells)]
        max_value = np.max(np.where(dists<dist_max))
        cells_to_keep = closest_cells[:max_value]
        # med = median_average_bw(cells_to_keep, R, pos)
        # print type (cells_to_keep)
        subset_dist = [np.mean([pos[cii] for cii in predecessor[ci]], axis=0) - pos[ci] for ci in cells_to_keep if not ci in R]
        if subset_dist != []:
            med_distance = spatial.distance.squareform(spatial.distance.pdist(subset_dist))
            med = subset_dist[np.argmin(np.sum(med_distance, axis=0))]
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

    gg_line = '\rGG done (%.2f s); '
    prop_line = 'P done (%.2f s); '
    SD_line = 'SD done (%.2f s); '
    fusion_line = 'F done (%.2f s); '
    nb_cells_line = '#C: %d'

    full_length = len(gg_line) + len(prop_line) + len(SD_line) + len(fusion_line) + len(nb_cells_line)

    
    for t in range(t_b, t_e, -1):
        # if t!=t_b and t%10 == 0:
        #     print t
        # else:
        #     print t,
        tic = time()
        print t, ': ',
        to_check_VF = LT.VF.time_nodes[t]

        idx3d, to_check_LT = LT.get_idx3d(t)

        LT.VF.time_nodes[t-1] = []
        mapping = []
        # tmp = np.array(to_check_LT)

        Gabriel_graph = LT.get_gabriel_graph(t)

        sys.stdout.write('\b'*(full_length) + ' '*(full_length))
        sys.stdout.flush()

        sys.stdout.write(gg_line%(time() - tic))
        sys.stdout.flush()

        tagged = {}
        cell_mapping_LT_VF = {}
        for C in to_check_VF:
            C_LT = to_check_LT[idx3d.query(LT.VF.pos[C])[1]]
            cell_mapping_LT_VF.setdefault(C_LT, []).append(C)
            # mapping += [(C, idx3d, nb_max, dist_max, tmp, LT.roots, LT.pos, LT.VF.pos, LT.successor, LT.LT.predecessor)]
            if not tagged.get(C_LT, False):
                mapping += [(C_LT, np.array(list(Gabriel_graph.get(C_LT, []))), dist_max)]
                tagged[C_LT] = True

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
                LT.VF.add_node(t-1, C_VF, LT.VF.pos[C_VF] + med)
            # C_next = LT.VF.get_next_id()
            # # C_next = CellSS(unique_id, unique_id, M=C, time = t-1, pos= C.pos + med)
            # LT.VF.time_nodes[t-1].append(C_next)
            # LT.VF.successor.setdefauLT(C, []).append(C_next)
            # LT.VF.edges.append((C, C_next))
            # LT.VF.nodes.append(C_next)
            # LT.VF.pos[C_next] = LT.VF.pos[C] + med

        idx3d, to_check_LT = LT.get_idx3d(t-1)
        to_check_VF = LT.VF.time_nodes[t-1]

        sys.stdout.write(prop_line%(time() - tic))
        sys.stdout.flush()

        if not LT.spatial_density.has_key(to_check_LT[0]):
            LT.compute_spatial_density(t-1, t-1, nb_max)

        sys.stdout.write(SD_line%(time() - tic))
        sys.stdout.flush()


        idx3d, to_check_VF = LT.VF.get_idx3d(t-1)[:2]
        dist_to_VF, equivalence = idx3d.query([LT.pos[c] for c in to_check_LT], 1)
        tmp = np.array([dist_to_VF[i]/LT.spatial_density[c] for i, c in enumerate(to_check_LT)])
        to_add = [to_check_LT[i] for i in np.where(tmp>1.25)[0]]
        for C in to_add:
            LT.VF.add_node(t-1, None, LT.pos[C])

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

        sys.stdout.write(fusion_line%(time() - tic))
        sys.stdout.flush()
        
        sys.stdout.write(nb_cells_line%i)
        sys.stdout.flush()
        sys.stdout.write("\n")
        sys.stdout.flush()

        # if len([k for k, s in LT.VF.successor.iteritems() if len(s)>1]) > 0:
        #     print 'oupsies t:', t


    LT.VF.t_b = t_b
    LT.VF.t_e = t_e

    return LT.VF


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
                if len(idx3d.query_ball_point((data[tmp_pt] + pos_e1)/2., (LT._dist_v(pos_e1, data[tmp_pt])/2)-10**-1))==0:
                    Gabriel_graph.setdefault(nodes[e1], set()).add(nodes[tmp_pt])
                    Gabriel_graph.setdefault(nodes[tmp_pt], set()).add(nodes[e1])
                if sorted_neighbs != []:
                    tmp_pt = sorted_neighbs.pop(0)
                else:
                    tmp_pt = None
    else:
        Gabriel_graph = LT.Gabriel_graph[t]

    return t, Gabriel_graph


def get_gabriel_graph(LT, t):
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
            while not tmp_pt is None and len(idx3d.query_ball_point((data[tmp_pt] + pos_e1)/2., (LT._dist_v(pos_e1, data[tmp_pt])/2)-10**-3))<=2:
                Gabriel_graph.setdefault(nodes[e1], set()).add(nodes[tmp_pt])
                Gabriel_graph.setdefault(nodes[tmp_pt], set()).add(nodes[e1])
                if sorted_neighbs != []:
                    tmp_pt = sorted_neighbs.pop(0)
                else:
                    tmp_pt = None

        LT.Gabriel_graph[t] = Gabriel_graph

    return LT.Gabriel_graph[t]

def parallel_gabriel_graph_preprocess(LT, nb_proc = 20):
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

def single_cell_propagation(params):
    C, closest_cells, dist_max = params
    # closest_cells = np.array(list(Gabriel_graph[C]))
    dists = np.array([np.sum((LT.pos[C]-LT.pos[n])**2)**.5 for n in closest_cells])
    if (dists<dist_max).any():
        cells_to_keep = closest_cells[np.where(dists<dist_max)]
        subset_dist = [np.mean([LT.pos[cii] for cii in LT.predecessor[ci]], axis=0) - LT.pos[ci] for ci in cells_to_keep if not LT.is_root[ci]]
        if subset_dist != []:
            med_distance = spatial.distance.squareform(spatial.distance.pdist(subset_dist))
            med = subset_dist[np.argmin(np.sum(med_distance, axis=0))]
        else:
            med = np.array([0, 0, 0])
    else:
        med = np.array([0, 0, 0])
    return C, med

class lineageTree(object):
    """docstring for lineageTree"""


    def _dist_v(self, v1, v2):
        v1 = np.array(v1)
        v2 = np.array(v2)
        return np.sum((v1-v2)**2)**(.5)

    def copy_cell(self, C, links=[]):
        C_tmp = copy(C)
        self.nodes.append(C)

    def get_next_id(self):
        if self.next_id == []:
            self.max_id += 1
            return self.max_id
        else:
            return self.next_id.pop()

    def add_node(self, t, succ, pos, id = None, reverse = False):
        if id is None:
            C_next = self.get_next_id()
        else:
            C_next = id
        self.time_nodes.setdefault(t, []).append(C_next)
        if not succ is None and not reverse:
            self.successor.setdefault(succ, []).append(C_next)
            self.predecessor.setdefault(C_next, []).append(succ)
            self.edges.append((succ, C_next))
        elif not succ is None:
            self.predecessor.setdefault(succ, []).append(C_next)
            self.successor.setdefault(C_next, []).append(succ)
            self.edges.append((C_next, succ))
        else:
            self.roots.append(C_next)
        self.nodes.append(C_next)
        self.pos[C_next] = pos
        self.progeny[C_next] = 0
        self.time[C_next] = t
        return C_next

    def remove_node(self, c):
        self.nodes.remove(c)
        self.time_nodes[self.time[c]].remove(c)
        # self.time_nodes.pop(c, 0)
        pos = self.pos.pop(c, 0)
        e_to_remove = [e for e in self.edges if c in e]
        for e in e_to_remove:
            self.edges.remove(e)
        if c in self.roots:
            self.roots.remove(c)
        succ = self.successor.pop(c, [])
        s_to_remove = [s for s, ci in self.successor.iteritems() if c in ci]
        for s in s_to_remove:
            self.successor[s].remove(c)

        pred = self.predecessor.pop(c, [])
        p_to_remove = [s for s, ci in self.predecessor.iteritems() if ci == c]
        for s in p_to_remove:
            self.predecessor[s].remove(c)

        self.time.pop(c, 0)
        self.spatial_density.pop(c, 0)

        self.next_id.append(c)
        return e_to_remove, succ, s_to_remove, pred, p_to_remove, pos

    def fuse_nodes(self, c1, c2):
        e_to_remove, succ, s_to_remove, pred, p_to_remove, c2_pos = self.remove_node(c2)
        for e in e_to_remove:
            new_e = [c1] + [other_c for other_c in e if e != c2]
            self.edges.append(new_e)

        self.successor.setdefault(c1, []).extend(succ)
        self.predecessor.setdefault(c1, []).extend(pred)

        for s in s_to_remove:
            self.successor[s].append(c1)

        for p in p_to_remove:
            self.predecessor[p].append(c1)


        self.pos[c1] = np.mean([self.pos[c1], c2_pos], axis = 0)
        self.progeny[c1] += 1

    def to_tlp(self, fname, t_min=-1, t_max=np.inf, temporal=True, spatial=False, VF=False):
        """
        Write a lineage tree into an understable tulip file
        fname : path to the tulip file to create
        lin_tree : lineage tree to write
        properties : dictionary of properties { 'Property name': [{c_id: prop_val}, default_val]}
        """
        
        f=open(fname, "w")

        f.write("(tlp \"2.0\"\n")
        f.write("(nodes ")
        if t_max!=np.inf or t_min>-1:
            nodes_to_use = [n for n in self.nodes if t_min<n.time<=t_max]
            edges_to_use = []
            if temporal:
                edges_to_use += [e for e in self.edges if t_min<e[0].time<t_max]
            if spatial:
                edges_to_use += [e for e in self.spatial_edges if t_min<e[0].time<t_max]
        else:
            nodes_to_use = self.nodes
            edges_to_use = []
            if temporal:
                edges_to_use += self.edges
            if spatial:
                edges_to_use += self.spatial_edges

        for n in nodes_to_use:
            f.write(str(n)+ " ")
        f.write(")\n")

        for i, e in enumerate(edges_to_use):
            f.write("(edge " + str(i) + " " + str(e[0]) + " " + str(e[1]) + ")\n")
        # f.write("(property 0 int \"id\"\n")
        # f.write("\t(default \"0\" \"0\")\n")
        # for n in nodes_to_use:
        #     f.write("\t(node " + str(n) + str(" \"") + str(self.n) + "\")\n")
        # f.write(")\n")

        f.write("(property 0 int \"time\"\n")
        f.write("\t(default \"0\" \"0\")\n")
        for n in nodes_to_use:
            f.write("\t(node " + str(n) + str(" \"") + str(self.time[n]) + "\")\n")
        f.write(")\n")

        f.write("(property 0 layout \"viewLayout\"\n")
        f.write("\t(default \"(0, 0, 0)\" \"()\")\n")
        for n in nodes_to_use:
            f.write("\t(node " + str(n) + str(" \"") + str(tuple(self.pos[n])) + "\")\n")
        f.write(")\n")

        f.write("(property 0 double \"distance\"\n")
        f.write("\t(default \"0\" \"0\")\n")
        for i, e in enumerate(edges_to_use):
            d_tmp = self._dist_v(self.pos[e[0]], self.pos[e[1]])
            f.write("\t(edge " + str(i) + str(" \"") + str(d_tmp) + "\")\n")
            f.write("\t(node " + str(e[0]) + str(" \"") + str(d_tmp) + "\")\n")
        f.write(")\n")

        # for property in properties:
        #     prop_name=property[0]
        #     vals=property[1]
        #     default=property[2]
        #     f.write("(property 0 string \""+prop_name+"\"\n")
        #     f.write("\t(default \""+str(default)+"\" \"0\")\n")
        #     for node in nodes:
        #         f.write("\t(node " + str(node) + str(" \"") + str(vals.get(node, default)) + "\")\n")
        #     f.write(")\n") 
        f.write(")")
        f.close()

    def median_average(self, subset):
        subset_dist = [np.mean([di.pos for di in c.D], axis = 0) - c.pos for c in subset if c.D != []]
        target_C = [c for c in subset if c.D != []]
        if subset_dist != []:
            med_distance = spatial.distance.squareform(spatial.distance.pdist(subset_dist))
            return subset_dist[np.argmin(np.sum(med_distance, axis=0))]
        else:
            return [0, 0, 0]

    def median_average_bw(self, subset):
        subset_dist = [c.M.pos - c.pos for c in subset if c.M != self.R]
        target_C = [c for c in subset if c.D != []]
        if subset_dist != []:
            med_distance = spatial.distance.squareform(spatial.distance.pdist(subset_dist))
            return subset_dist[np.argmin(np.sum(med_distance, axis=0))]
        else:
            return [0, 0, 0]

    def build_median_vector(self, C, dist_th, delta_t = 2):#temporal_space=lambda d, t, c: d+(t*c)):
        if not hasattr(self, 'spatial_edges'):
            self.compute_spatial_edges(dist_th)
        subset = [C]
        subset += C.N
        added_D = added_M = subset
        for i in xrange(delta_t):
            _added_D = []
            _added_M = []
            for c in added_D:
                _added_D += c.D
            for c in added_M:
                if not c.M is None:
                    _added_M += [c.M]
            subset += _added_M
            subset += _added_D
            added_D = _added_D
            added_M = _added_M


        return self.median_average(subset)

    def build_vector_field(self, dist_th=50):
        ruler = 0
        for C in self.nodes:
            if ruler != C.time:
                print C.time
            C.direction = self.build_median_vector(C, dist_th)
            ruler = C.time
    
    def single_cell_propagation(self, params):
        C, t, nb_max, dist_max, to_check_self, R, pos, successor, predecessor = params
        idx3d = self.kdtrees[t]
        closest_cells = np.array(to_check_self)[list(idx3d.query(tuple(pos[C]), nb_max)[1])]
        max_value = np.min(np.where(np.array([_dist_v(pos[C], pos[ci]) for ci in closest_cells]+[dist_max+1])>dist_max))
        cells_to_keep = closest_cells[:max_value]
        # med = median_average_bw(cells_to_keep, R, pos)
        # print type (cells_to_keep)
        subset_dist = [np.mean([pos[cii] for cii in predecessor[ci]], axis=0) - pos[ci] for ci in cells_to_keep if not ci in R]
        if subset_dist != []:
            med_distance = spatial.distance.squareform(spatial.distance.pdist(subset_dist))
            med = subset_dist[np.argmin(np.sum(med_distance, axis=0))]
        else:
            med = [0, 0, 0]
        return C, med
    
    def read_from_xml(self, file_format, tb, te, z_mult=1., mask = None):
        self.time_nodes = {}
        self.time_edges = {}
        unique_id = 0
        self.nodes = []
        self.edges = []
        self.roots = []
        self.successor = {}
        self.predecessor = {}
        self.pos = {}
        self.time_id = {}
        self.time = {}
        self.mother_not_found = []
        self.ind_cells = {}
        self.svIdx = {}
        self.is_root = {}
        self.lin = {}
        self.C_lin = {}
        self.coeffs = {}
        self.intensity = {}
        self.W = {}
        for t in range(tb, te+1):
            print t,
            if t%10==0:
                print
            tree = ET.parse(file_format%t)
            root = tree.getroot()
            self.time_nodes[t] = []
            self.time_edges[t] = []
            for it in root:
                if not '-1.#IND' in it.attrib['m']:
                    M_id, pos, cell_id, svIdx, lin_id = (int(it.attrib['parent']), 
                                                [float(v) for v in it.attrib['m'].split(' ') if v!=''], 
                                                int(it.attrib['id']),
                                                [int(v) for v in it.attrib['svIdx'].split(' ') if v!=''],
                                                int(it.attrib['lineage']))
                    try:
                        alpha, W, nu, alphaPrior = (float(it.attrib['alpha']),
                                                    [float(v) for v in it.attrib['W'].split(' ') if v!=''],
                                                    float(it.attrib['nu']),
                                                    float(it.attrib['alphaPrior']))
                        pos = np.array(pos)
                        pos_tmp = np.round(pos).astype(np.uint16)
                        if mask is None or mask[pos_tmp[0], pos_tmp[1], pos_tmp[2]]:
                            C = unique_id
                            pos[-1] = pos[-1]*z_mult
                            if (t-1, M_id) in self.time_id:
                                M = self.time_id[(t-1, M_id)]
                                self.successor.setdefault(M, []).append(C)
                                self.predecessor.setdefault(C, []).append(M)
                                self.edges.append((M, C))
                                self.time_edges[t].append((M, C))
                                self.is_root[C] = False
                            else:
                                if M_id != -1:
                                    self.mother_not_found.append(C)
                                self.roots.append(C)
                                self.is_root[C] = True
                            self.pos[C] = pos
                            self.nodes.append(C)
                            self.time_nodes[t].append(C)
                            self.time_id[(t, cell_id)] = C
                            self.time[C] = t
                            self.svIdx[C] = svIdx
                            self.lin.setdefault(lin_id, []).append(C)
                            self.C_lin[C] = lin_id
                            self.intensity[C] = max(alpha - alphaPrior, 0)
                            tmp = list(np.array(W) * nu)
                            self.W[C] = np.array(W).reshape(3, 3)
                            self.coeffs[C] = tmp[:3] + tmp[4:6] + tmp[8:9] + list(pos)
                            unique_id += 1
                    except Exception, e:
                        pass
                else:
                    if self.ind_cells.has_key(t):
                        self.ind_cells[t] += 1
                    else:
                        self.ind_cells[t] = 1
        self.max_id = unique_id - 1

    def read_from_mamut_xml(self, path):
        tree = ET.parse(path)
        Model = tree.getroot()[0]
        FeatureDeclarations, AllSpots, AllTracks, FilteredTracks = list(Model)

        self.time_nodes = {}
        self.time_edges = {}
        self.nodes = []
        self.pos = {}
        self.time = {}
        self.node_name = {}
        for frame in AllSpots:
            t = int(frame.attrib['frame'])
            self.time_nodes[t] = []
            for cell in frame:
                cell_id, n, x, y, z = (int(cell.attrib['ID']), cell.attrib['name'],
                                                 float(cell.attrib['POSITION_X']),
                                                 float(cell.attrib['POSITION_Y']),
                                                 float(cell.attrib['POSITION_Z']))
                self.time_nodes[t].append(cell_id)
                self.nodes.append(cell_id)
                self.pos[cell_id] = np.array([x, y, z])
                self.time[cell_id] = t
                self.node_name[cell_id] = n

        self.edges = []
        self.roots = []
        tracks = {}
        self.successor = {}
        self.predecessor = {}
        for track in AllTracks:
            t_id, l = int(track.attrib['TRACK_ID']), float(track.attrib['TRACK_DURATION'])
            tracks[t_id] = []
            for edge in track:
                s, t = int(edge.attrib['SPOT_SOURCE_ID']), int(edge.attrib['SPOT_TARGET_ID'])
                if s in self.nodes and t in self.nodes:
                    if self.time[s] > self.time[t]:
                        s, t = t, s
                    self.successor.setdefault(s, []).append(t)
                    self.predecessor.setdefault(t, []).append(s)
                    tracks[t_id].append((s, t))
                    self.edges.append((s, t))
        self.t_b = min(self.time_nodes.keys())
        self.t_e = max(self.time_nodes.keys())
    
    def to_binary(self, fname, starting_points = None):    
        if starting_points is None:
            starting_points = [c for c in self.successor.iterkeys() if self.predecessor.get(c, []) != []]
        number_sequence = [-1]
        pos_sequence = []
        time_sequence = []
        default_lin = -1
        for c in starting_points:
            time_sequence.append(self.time.get(c, 0))
            to_treat = [c]
            while to_treat != []:
                curr_c = to_treat.pop()
                number_sequence.append(curr_c)
                pos_sequence += list(self.pos[curr_c])
                if not curr_c in self.successor:
                    number_sequence.append(-1)
                elif len(self.successor[curr_c]) == 1:
                    to_treat += self.successor[curr_c]
                else:
                    number_sequence.append(-2)
                    to_treat += self.successor[curr_c]

        remaining_nodes = set(self.nodes) - set(number_sequence)

        for c in remaining_nodes:
            time_sequence.append(self.time.get(c, 0))
            number_sequence.append(c)
            pos_sequence += list(self.pos[c])
            number_sequence.append(-1)

        f = open(fname, 'wb')
        f.write(struct.pack('q', len(number_sequence)))
        f.write(struct.pack('q', len(time_sequence)))
        f.write(struct.pack('q', len(pos_sequence)))
        f.write(struct.pack('q'*len(number_sequence), *number_sequence))
        f.write(struct.pack('H'*len(time_sequence), *time_sequence))
        f.write(struct.pack('d'*len(pos_sequence), *pos_sequence))

        f.close()


    def read_from_binary(self, fname, reverse_time = False):
        q_size = struct.calcsize('q')
        H_size = struct.calcsize('H')
        d_size = struct.calcsize('d')

        f = open(fname, 'rb')
        len_tree = struct.unpack('q', f.read(q_size))[0]
        len_time = struct.unpack('q', f.read(q_size))[0]
        len_pos = struct.unpack('q', f.read(q_size))[0]
        number_sequence = list(struct.unpack('q'*len_tree, f.read(q_size*len_tree)))
        time_sequence = list(struct.unpack('H'*len_time, f.read(H_size*len_time)))
        pos_sequence = np.array(struct.unpack('d'*len_pos, f.read(d_size*len_pos)))

        f.close()

        successor = {}
        predecessor = {}
        time = {}
        time_nodes = {}
        time_edges = {}
        pos = {}
        is_root = {}
        nodes = []
        edges = []
        waiting_list = []
        print number_sequence[0]
        for i, c in enumerate(number_sequence[:-1]):
            if c == -1:
                if waiting_list != []:
                    prev_mother = waiting_list.pop()
                    successor[prev_mother].insert(0, number_sequence[i+1])
                    edges.append((prev_mother, number_sequence[i+1]))
                    time_edges.setdefault(t, []).append((prev_mother, number_sequence[i+1]))
                    is_root[number_sequence[i+1]] = False
                    t = time[prev_mother] + 1
                else:
                    t = time_sequence.pop(0)
                    is_root[number_sequence[i+1]] = True

            elif c == -2:
                successor[waiting_list[-1]] = [number_sequence[i+1]]
                edges.append((waiting_list[-1], number_sequence[i+1]))
                time_edges.setdefault(t, []).append((waiting_list[-1], number_sequence[i+1]))
                is_root[number_sequence[i+1]] = False
                pos[waiting_list[-1]] = pos_sequence[:3]
                pos_sequence = pos_sequence[3:]
                nodes.append(waiting_list[-1])
                time[waiting_list[-1]] = t
                time_nodes.setdefault(t, []).append(waiting_list[-1])
                t += 1

            elif number_sequence[i+1] >= 0:
                successor[c] = [number_sequence[i+1]]
                edges.append((c, number_sequence[i+1]))
                time_edges.setdefault(t, []).append((c, number_sequence[i+1]))
                is_root[number_sequence[i+1]] = False
                pos[c] = pos_sequence[:3]
                pos_sequence = pos_sequence[3:]
                nodes.append(c)
                time[c] = t
                time_nodes.setdefault(t, []).append(c)
                t += 1

            elif number_sequence[i+1] == -2:
                waiting_list += [c]

            elif number_sequence[i+1] == -1:
                pos[c] = pos_sequence[:3]
                pos_sequence = pos_sequence[3:]
                nodes.append(c)
                time[c] = t
                time_nodes.setdefault(t, []).append(c)
                t += 1

        predecessor = {vi: [k] for k, v in successor.iteritems() for vi in v}

        self.successor = successor
        self.predecessor = predecessor
        self.time = time
        self.time_nodes = time_nodes
        self.time_edges = time_edges
        self.pos = pos
        self.nodes = nodes
        self.edges = edges
        self.t_b = min(time_nodes.iterkeys())
        self.t_e = max(time_nodes.iterkeys())
        self.is_root = is_root
        self.max_id = max(self.nodes)
    
    def write_to_prune(self, file_format_input, file_format_output):
        old_id_to_new = {}
        old_lin_to_new = {}
        lin_id = 0
        for t in range(200, 205+1):
            new_id = 0
            print t,
            if t%10==0:
                print
            tree = ET.parse(file_format_input%t)
            root = tree.getroot()
            for it in list(root.getchildren()):
                if not '-1.#IND' in it.attrib['m']:
                    M_id, pos, cell_id, old_lin_id = (int(it.attrib['parent']), 
                                                   [float(v) for v in it.attrib['m'].split(' ') if v!=''], 
                                                   int(it.attrib['id']), int(it.attrib['lineage']))
                    if not self.to_keep.get(self.time_id[(t, cell_id)], False):
                        root.remove(it)
                    else:

                        # if t == 201 and old_lin_to_new.get(old_lin_id, 0) == 0:#old_id_to_new.get((t-1, M_id), 0) == 12:
                            # print 'YO'
                            # break
                        # break
                        if M_id != -1:
                            it.set('parent', M_id)#str(old_id_to_new[(t-1, M_id)]))
                        it.set('id', cell_id)#str(new_id))
                        old_id_to_new[(t, cell_id)] = new_id
                        new_id += 1
                        # if not old_lin_to_new.has_key(old_lin_id):
                        #     old_lin_to_new[old_lin_id] = lin_id
                        #     lin_id += 1
                        # it.set('lineage', str(old_lin_to_new[old_lin_id]))
            tree.write(file_format_output%t)

    def get_idx3d(self, t):
        to_check_self = self.time_nodes[t]
        if not self.kdtrees.has_key(t):
            data_corres = {}
            data = []
            for i, C in enumerate(to_check_self):
                data.append(tuple(self.pos[C]))
                data_corres[i] = C
            idx3d = kdtree.KDTree(data)
            self.kdtrees[t] = idx3d
        else:
            idx3d = self.kdtrees[t]
        return idx3d, to_check_self

    def get_gabriel_graph(self, t):
        if not hasattr(self, 'Gabriel_graph'):
            self.Gabriel_graph = {}

        if not self.Gabriel_graph.has_key(t):
            idx3d, nodes = self.get_idx3d(t)

            data_corres = {}
            data = []
            for i, C in enumerate(nodes):
                data.append(self.pos[C])
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
                distances = np.array([self._dist_v(pos_e1, data[e2]) for e2 in ni])
                sorted_neighbs = list(np.array(ni)[np.argsort(distances)])
                tmp_pt = sorted_neighbs.pop(0)
                while not tmp_pt is None and len(idx3d.query_ball_point((data[tmp_pt] + pos_e1)/2., (self._dist_v(pos_e1, data[tmp_pt])/2)-10**-3))<=2:
                    Gabriel_graph.setdefault(nodes[e1], set()).add(nodes[tmp_pt])
                    Gabriel_graph.setdefault(nodes[tmp_pt], set()).add(nodes[e1])
                    if sorted_neighbs != []:
                        tmp_pt = sorted_neighbs.pop(0)
                    else:
                        tmp_pt = None

            self.Gabriel_graph[t] = Gabriel_graph

        return self.Gabriel_graph[t]

    def parallel_gabriel_graph_preprocess(self, nb_proc = 20):
        mapping = []
        if not hasattr(self, 'Gabriel_graph'):
            self.Gabriel_graph = {}
        for t in xrange(self.t_b, self.t_e + 1):
            if not self.Gabriel_graph.has_key(t):
                mapping += [(self, t)]
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
            self.Gabriel_graph[t] = G_g


    def build_VF_propagation_backward(self, t_b=0, t_e=200, nb_max=20, dist_max=200, nb_proc = 8):
        self.VF = lineageTree(None, None, None)
        from time import time

        # Hack to allow pickling of kdtrees for multiprocessing
        kdtree.node = kdtree.KDTree.node
        kdtree.leafnode = kdtree.KDTree.leafnode
        kdtree.innernode = kdtree.KDTree.innernode

        starting_cells = self.time_nodes[t_b]
        unique_id = 0
        self.VF.time_nodes = {t_b: []}
        for C in starting_cells:
            # C_tmp = CellSS(unique_id=unique_id, id=unique_id, M=self.VF.R, time = t_b, pos = C.pos)
            i = self.VF.get_next_id()
            self.VF.nodes.append(i)
            self.VF.time_nodes[t_b].append(i)
            self.VF.roots.append(i)
            self.VF.pos[i]=self.pos[C]

        for t in range(t_b, t_e, -1):
            # if t!=t_b and t%10 == 0:
            #     print t
            # else:
            #     print t,
            tic = time()
            print t, ': ',
            to_check_VF = self.VF.time_nodes[t]

            idx3d, to_check_self = self.get_idx3d(t)

            self.VF.time_nodes[t-1] = []
            mapping = []
            # tmp = np.array(to_check_self)

            Gabriel_graph = self.get_gabriel_graph(t)

            print 'Gabriel graph built (%f s); '%(time() - tic),

            cell_mapping_LT_VF = {}
            for C in to_check_VF:
                C_self = to_check_self[idx3d.query(self.VF.pos[C])[1]]
                cell_mapping_LT_VF[C_self] = C
                # mapping += [(C, idx3d, nb_max, dist_max, tmp, self.roots, self.pos, self.VF.pos, self.successor, self.predecessor)]
                mapping += [(C_self, Gabriel_graph, dist_max, self.roots, self.pos, self.predecessor)]

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
            for C, med in out:
                C_VF = cell_mapping_LT_VF[C]
                self.VF.add_node(t-1, C_VF, self.VF.pos[C_VF] + med)
                # C_next = self.VF.get_next_id()
                # # C_next = CellSS(unique_id, unique_id, M=C, time = t-1, pos= C.pos + med)
                # self.VF.time_nodes[t-1].append(C_next)
                # self.VF.successor.setdefauself(C, []).append(C_next)
                # self.VF.edges.append((C, C_next))
                # self.VF.nodes.append(C_next)
                # self.VF.pos[C_next] = self.VF.pos[C] + med

            idx3d, to_check_self = self.get_idx3d(t-1)
            to_check_VF = self.VF.time_nodes[t-1]

            print 'propagation done (%f s); '%(time() - tic),

            if not self.spatial_density.has_key(to_check_self[0]):
                self.compute_spatial_density(t-1, t-1, nb_max)

            print 'Spatial density done (%f s); '%(time() - tic),

            dist_to_VF, equivalence = idx3d.query([self.VF.pos[c] for c in to_check_VF], 1)
            # equivalence = equivalence[:,1]

            count = np.bincount(equivalence)
            self_too_close, = np.where(count > 1)
            TMP = []
            for C_self in self_too_close:
                to_potentially_fuse, = np.where(equivalence == C_self)
                pos_tmp = [self.VF.pos[to_check_VF[c]] for c in to_potentially_fuse]
                dist_tmp = spatial.distance.squareform(spatial.distance.pdist(pos_tmp))
                dist_tmp[dist_tmp==0] = np.inf
                if (dist_tmp<self.spatial_density[to_check_self[C_self]]/2.).any():
                    to_fuse = np.where(dist_tmp == np.min(dist_tmp))[0]
                    c1, c2 = to_potentially_fuse[list(to_fuse)][:2]
                    to_check_VF[c1], to_check_VF[c2]
                    TMP.append([to_check_VF[c1], to_check_VF[c2]])

            for c1, c2 in TMP:
                # new_pos = np.mean([self.VF.pos[c1], self.VF.pos[c2]], axis = 0)
                if c1 != c2:
                    self.VF.fuse_nodes(c1, c2)

            print 'Fusion done (%f s); '%(time() - tic),

            idx3d, to_check_VF = self.VF.get_idx3d(t-1)[:2]
            dist_to_VF, equivalence = idx3d.query([self.pos[c] for c in to_check_self], 1)
            tmp = np.array([dist_to_VF[i]/self.spatial_density[c] for i, c in enumerate(to_check_self)])
            to_add = [to_check_self[i] for i in np.where(tmp>1)[0]]
            for C in to_add:
                self.VF.add_node(t-1, None, self.pos[C])

            print 'Addition done (%f s); '%(time() - tic),
            print '#cells: %d'%(len(self.VF.time_nodes[t-1]))

        self.VF.t_b = t_b
        self.VF.t_e = t_e

        return self.VF

    def compute_spatial_density(self, t_b=0, t_e=200, n_size=10):
        time_range = [t for t in self.time_nodes.keys() if t_b <= t <= t_e]
        for t in time_range:
            Cs = self.time_nodes[t]
            data_corres = {}
            data = []
            for i, C in enumerate(Cs):
                data.append(tuple(self.pos[C]))
                data_corres[i] = C
            if not self.kdtrees.has_key(t):
                idx3d = kdtree.KDTree(data)
                # self.kdtrees[t] = idx3d
            else:
                idx3d = self.kdtrees[t]
            distances, indices = idx3d.query(data, n_size)
            self.spatial_density.update(dict(zip(Cs, np.mean(distances[:, 1:], axis=1))))

    def compute_spatial_edges(self, th=50):
        self.spatial_edges=[]
        for t, Cs in self.time_nodes.iteritems():
            nodes_tmp, pos_tmp = zip(*[(C, C.pos) for C in Cs])
            nodes_tmp = np.array(nodes_tmp)
            distances = spatial.distance.squareform(spatial.distance.pdist(pos_tmp))
            nodes_to_match = np.where((0<distances) & (distances<th))
            to_link = zip(nodes_tmp[nodes_to_match[0]], nodes_tmp[nodes_to_match[1]])
            self.spatial_edges.extend(to_link)
            for C1, C2 in to_link:
                C1.N.append(C2)

    def __init__(self, file_format, tb = None, te = None, z_mult = 1., mask = None, MaMuT = False):
        super(lineageTree, self).__init__()
        self.time_nodes = {}
        self.time_edges = {}
        self.max_id = -1
        self.next_id = []
        self.nodes = []
        self.edges = []
        self.roots = []
        self.successor = {}
        self.predecessor = {}
        self.pos = {}
        self.time_id = {}
        self.time = {}
        self.kdtrees = {}
        self.spatial_density = {}
        self.progeny = {}
        if not (file_format is None or tb is None or te is None) and not MaMuT:
            self.read_from_xml(file_format, tb, te, z_mult=z_mult, mask = mask)
            self.t_b = tb
            self.t_e = te
        elif not (file_format is None) and MaMuT:
            self.read_from_mamut_xml(file_format)
        elif not (file_format is None):
            self.read_from_binary(file_format)
