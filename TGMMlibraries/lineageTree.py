#!python
# This file is subject to the terms and conditions defined in
# file 'LICENCE', which is part of this source code package.
# Author: Leo Guignard (guignardl...@AT@...janelia.hhmi.org)

from scipy.spatial import cKDTree as KDTree
import os
import xml.etree.ElementTree as ET
from copy import copy
from scipy import spatial
import numpy as np
from multiprocessing import Pool
from scipy.spatial import Delaunay
from itertools import combinations
from numbers import Number
import struct
import sys

class lineageTree(object):
    ''' lineageTree is a class container for lineage tree structures
        The main attributes are the following:
        self.nodes: [int, ], list of node/cell ids
        self.edges: [(int, int), ], a list of couple of cell/objects ids that represents the edges
        self.time_nodes: {int, [int, ]}, a dictionary that maps time points to
            a list of cell ids that belong to that time point
        self.time_edges: {int, [(int, int), ]}, a dictionary that maps time points to
            a list of edges couples that belong to that time point
        self.successor: {int, [int, ]}, a dictionary that maps a cell id to
            the list of its successors in time
        self.predecessor: {int, [int, ]}, a dictionary that maps a cell id to
            the list of its predecessors in time
        self.time: {int: int, }, a dictionary that maps a cell to the time
            it belongs to.
    '''

    def _dist_v(self, v1, v2):
        ''' Computes the L2 norm between two vectors v1 and v2
            Args:
                v1: [float, ], list of values for the first vector
                v2: [float, ], list of values for the second vector
            Return:
                float: L2 norm between v1 and v2
        '''
        v1 = np.array(v1)
        v2 = np.array(v2)
        return np.sum((v1-v2)**2)**(.5)

    def get_next_id(self):
        ''' Computes the next authorized id.
            Returns:
                int, next authorized id
        '''
        if self.next_id == []:
            self.max_id += 1
            return self.max_id
        else:
            return self.next_id.pop()

    def add_node(self, t=None, succ=None, pos=None, id=None, reverse=False):
        ''' Adds a node to the lineageTree and update it accordingly.
            Args:
                t: int, time to which to add the node
                succ: id of the node the new node is a successor to
                pos: [float, ], list of three floats representing the 3D spatial position of the node
                id: id value of the new node, to be used carefully,
                    if None is provided the new id is automatically computed.
                reverse: bool, True if in this lineageTree the predecessors are the successors and reciprocally.
                    This is there for bacward compatibility, should be left at False.
            Returns:
                C_next: int, id of the new node.
        '''
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
        ''' Removes a node and update the lineageTree accordingly
            Args:
                c: int, id of the node to remove
        '''
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
        s_to_remove = [s for s, ci in self.successor.items() if c in ci]
        for s in s_to_remove:
            self.successor[s].remove(c)

        pred = self.predecessor.pop(c, [])
        p_to_remove = [s for s, ci in self.predecessor.items() if ci == c]
        for s in p_to_remove:
            self.predecessor[s].remove(c)

        self.time.pop(c, 0)
        self.spatial_density.pop(c, 0)

        self.next_id.append(c)
        return e_to_remove, succ, s_to_remove, pred, p_to_remove, pos

    def fuse_nodes(self, c1, c2):
        ''' Fuses together two nodes that belong to the same time point
            and update the lineageTree accordingly.
            Args:
                c1: int, id of the first node to fuse
                c2: int, id of the second node to fuse
        '''
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


    def _write_header_am(self, f, nb_points, length):
        ''' Header for Amira .am files
        '''
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

    def write_to_am(self, path_format, t_b=None, t_e=None, length=5, manual_labels=None,
                      default_label=5, new_pos=None):
        ''' Writes a lineageTree into an Amira readable data (.am format).
            Args:
                path_format: string, path to the output. It should contain 1 %03d where the time step will be entered
                t_b: int, first time point to write (if None, min(LT.to_take_time) is taken)
                t_e: int, last time point to write (if None, max(LT.to_take_time) is taken)
                    note: if there is no 'to_take_time' attribute, self.time_nodes is considered instead
                        (historical)
                length: int, length of the track to print (how many time before).
                manual_labels: {id: label, }, dictionary that maps cell ids to
                default_label: int, default value for the manual label
                new_pos: {id: [x, y, z]}, dictionary that maps a 3D position to a cell ID.
                    if new_pos == None (default) then self.pos is considered.
        '''
        if not hasattr(self, 'to_take_time'):
            self.to_take_time = self.time_nodes
        if t_b is None:
            t_b = min(self.to_take_time.keys())
        if t_e is None:
            t_e = max(self.to_take_time.keys())
        if new_pos is None:
            new_pos = self.pos

        if manual_labels is None:
            manual_labels = {}
        for t in range(t_b, t_e + 1):
            f = open(path_format%t, 'w')
            nb_points = len(self.to_take_time[t])
            self._write_header_am(f, nb_points, length)
            points_v = {}
            for C in self.to_take_time[t]:
                C_tmp = C
                positions = []
                for i in range(length):
                    C_tmp = self.predecessor.get(C_tmp, [C_tmp])[0]
                    positions.append(new_pos[C_tmp])
                points_v[C] = positions

            f.write('@1\n')
            for i, C in enumerate(self.to_take_time[t]):
                f.write('%f %f %f\n'%tuple(points_v[C][0]))
                f.write('%f %f %f\n'%tuple(points_v[C][-1]))

            f.write('@2\n')
            for i, C in enumerate(self.to_take_time[t]):
                f.write('%d %d\n'%(2*i, 2*i+1))

            f.write('@3\n')
            for i, C in enumerate(self.to_take_time[t]):
                f.write('%d\n'%(length))

            f.write('@4\n')
            tmp_velocity = {}
            for i, C in enumerate(self.to_take_time[t]):
                for p in points_v[C]:
                    f.write('%f %f %f\n'%tuple(p))

            f.write('@5\n')
            for i, C in enumerate(self.to_take_time[t]):
                f.write('%f\n'%(manual_labels.get(C, default_label)))
                f.write('%f\n'%(0))

            f.write('@6\n')
            for i, C in enumerate(self.to_take_time[t]):
                f.write('%d\n'%(int(manual_labels.get(C, default_label) != default_label)))
                f.write('%d\n'%(0))

            f.write('@7\n')
            for i, C in enumerate(self.to_take_time[t]):
                f.write('%f\n'%(np.linalg.norm(points_v[C][0] - points_v[C][-1])))

            f.write('@8\n')
            for i, C in enumerate(self.to_take_time[t]):
                f.write('%d\n'%(1))
                f.write('%d\n'%(0))
            f.close()

    def get_height(self, c, done):
        if c in done:
            return done[c][0]
        else:
            P = np.mean([self.get_height(di, done) for di in self.successor[c]])
            done[c] = [P, self.vert_space_factor*self.time[c]]
            return P

    def print_to_SVG(self, file_name, roots=None, draw_nodes=True, draw_edges=True,
                     order_key=None, vert_space_factor=.5, horizontal_space=1,
                     node_size=None, stroke_width=None, factor=1.,
                     node_color=None, stroke_color=None, positions=None):
        ''' Writes the lineage tree to an SVG file.
            Node and edges coloring and size can be provided.
            Args:
                file_name: str, filesystem filename valid for `open()`
                roots: [int, ...], list of node ids to be drawn. If `None` all the nodes will be drawn. Default `None`
                draw_nodes: bool, wether to print the nodes or not, default `True`
                draw_edges: bool, wether to print the edges or not, default `True`
                order_key: function that would work for the attribute `key=` for the `sort`/`sorted` function
                vert_space_factor: float, the vertical position of a node is its time. `vert_space_factor` is a
                                   multiplier to space more or less nodes in time
                horizontal_space: float, space between two consecutive nodes
                node_size: func, a function that maps a node id to a `float` value that will determine the
                           radius of the node. The default function return the constant value `vertical_space_factor/2.1`
                stroke_width: func, a function that maps a node id to a `float` value that will determine the
                              width of the daughter edge.  The default function return the constant value `vertical_space_factor/2.1`
                factor: float, scaling factor for nodes positions, default 1
                node_color: func, a function that maps a node id to a triplet between 0 and 255.
                            The triplet will determine the color of the node.
                stroke_color: func, a function that maps a node id to a triplet between 0 and 255.
                              The triplet will determine the color of the stroke of the inward edge.
                positions: {int: [float, float], ...}, dictionary that maps a node id to a 2D position.
                           Default `None`. If provided it will be used to position the nodes.
        '''
        import svgwrite

        if roots is None:
            roots = list(set(self.successor).difference(self.predecessor))

        if node_size is None:
            node_size = lambda x: vert_space_factor/2.1
        if stroke_width is None:
            stroke_width = lambda x: vert_space_factor/2.2
        if node_color is None:
            node_color = lambda x: (0, 0, 0)
        coloring_edges = not stroke_color is None
        if not coloring_edges:
            stroke_color = lambda x: (0, 0, 0)
        prev_x = 0
        self.vert_space_factor = vert_space_factor
        if order_key is not None:
            roots.sort(key=order_key)
        treated_cells = []

        pos_given = not positions is None
        if not pos_given:
            positions = dict(zip(self.nodes, [[0., 0.],]*len(self.nodes)))
        for i, r in enumerate(roots):
            r_leaves = []
            to_do = [r]
            while len(to_do) != 0:
                curr = to_do.pop(0)
                treated_cells += [curr]
                if curr in self.successor:
                    if order_key is not None:
                        to_do += sorted(self.successor[curr], key=order_key)
                    else:
                        to_do += self.successor[curr]
                else:
                    r_leaves += [curr]
            r_pos = {l: [prev_x+horizontal_space*(1+j), self.vert_space_factor*self.time[l]]
                         for j, l in enumerate(r_leaves)}
            self.get_height(r, r_pos)
            prev_x = np.max(list(r_pos.values()), axis=0)[0]
            if not pos_given:
                positions.update(r_pos)

        dwg = svgwrite.Drawing(file_name, profile='tiny', size=factor*np.max(list(positions.values()), axis=0))
        if draw_edges and not draw_nodes and not coloring_edges:
            to_do = set(treated_cells)
            while 0<len(to_do):
                curr = to_do.pop()
                c_cycle = self.get_cycle(curr)
                x1 , y1 = positions[c_cycle[0]]
                x2 , y2 = positions[c_cycle[-1]]
                dwg.add(dwg.line((factor*x1, factor*y1), (factor*x2, factor*y2), stroke=svgwrite.rgb(0, 0, 0)))
                for si in self.successor.get(c_cycle[-1], []):
                    x3, y3 = positions[si]
                    dwg.add(dwg.line((factor*x2, factor*y2), (factor*x3, factor*y3), stroke=svgwrite.rgb(0, 0, 0)))
                to_do.difference_update(c_cycle)
        else:
            for c in treated_cells:
                x1, y1 = positions[c]
                for si in self.successor.get(c, []):
                    x2, y2 = positions[si]
                    if draw_edges:
                        dwg.add(dwg.line((factor*x1, factor*y1), (factor*x2, factor*y2),
                                stroke=svgwrite.rgb(*(stroke_color(si))),
                                stroke_width=svgwrite.pt(stroke_width(si))))
            for c in treated_cells:
                x1, y1 = positions[c]
                if draw_nodes:
                    dwg.add(dwg.circle((factor*x1, factor*y1), node_size(c), fill=svgwrite.rgb(*(node_color(c)))))
        dwg.save()

    def to_tlp(self, fname, t_min=-1, t_max=np.inf, nodes_to_use=None, temporal=True, spatial=False,
               VF=False, write_layout=True, node_properties=None, Names=False):
        '''
        Write a lineage tree into an understable tulip file
        Args:
            fname: string, path to the tulip file to create
            t_min: int, minimum time to consider, default -1
            t_max: int, maximum time to consider, default np.inf
            nodes_to_use: [int, ], list of nodes to show in the graph,
                          default *None*, then self.nodes is used
                          (taking into account *t_min* and *t_max*)
            temporal: boolean, True if the temporal links should be printed, default True
            spatial: boolean, True if the special links should be printed, default True
            VF: boolean, useless
            write_layout: boolean, True, write the spatial position as layout,
                                   False, do not write spatial positionm
                                   default True
            node_properties = {'p_name':[{id:p_value, }, default]}, a dictionary of properties to write
                                                To a key representing the name of the property is
                                                paired a dictionary that maps a cell id to a property
                                                and a default value for this property
        '''
        def format_names(names_which_matter):
            """
            Return an ensured formated cell names
            """
            tmp={}
            for k, v in names_which_matter.items():
                try:
                    val=v.split('.')[1][:-1]
                    tmp[k] = (v.split('.')[0][0] + '%02d'%int(v.split('.')[0][1:]) + '.'
                              + '%04d'%int(v.split('.')[1][:-1]) + v.split('.')[1][-1])
                except Exception as e:
                    print(v)
                    exit()
            return tmp

        f=open(fname, "w")
        f.write("(tlp \"2.0\"\n")
        f.write("(nodes ")

        if not nodes_to_use:
            if t_max!=np.inf or -1<t_min:
                nodes_to_use = [n for n in self.nodes if t_min<self.time[n]<=t_max]
                edges_to_use = []
                if temporal:
                    edges_to_use += [e for e in self.edges if t_min<self.time[e[0]]<t_max]
                if spatial:
                    edges_to_use += [e for e in self.spatial_edges if t_min<self.time[e[0]]<t_max]
            else:
                nodes_to_use = list(self.nodes)
                edges_to_use = []
                if temporal:
                    edges_to_use += self.edges
                if spatial:
                    edges_to_use += self.spatial_edges
        else:
            edges_to_use = []
            if temporal:
                edges_to_use += [e for e in self.edges if e[0] in nodes_to_use and e[1] in nodes_to_use]
            if spatial:
                edges_to_use += [e for e in self.spatial_edges if t_min<e[0].time<t_max]
        nodes_to_use = set(nodes_to_use)
        if Names:
            names_which_matter = { k : v for k, v in node_properties[Names][0].items()
                                     if v!='' and v!='NO' and k in nodes_to_use}
            names_formated = format_names(names_which_matter)
            order_on_nodes = np.array(list(names_formated.keys()))[np.argsort(list(names_formated.values()))]
            nodes_to_use = set(nodes_to_use).difference(order_on_nodes)
            tmp_names = {}
            for k, v in node_properties[Names][0].items():
                if len(self.successor.get(self.predecessor.get(k, [-1])[0], [])) != 1 or self.time[k]==t_min+1:
                    tmp_names[k] = v
            node_properties[Names][0] = tmp_names
            for n in order_on_nodes:
               f.write(str(n)+ " ")
        else:
            order_on_nodes = set()

        for n in nodes_to_use:
            f.write(str(n)+ " ")
        f.write(")\n")

        nodes_to_use.update(order_on_nodes)

        for i, e in enumerate(edges_to_use):
            f.write("(edge " + str(i) + " " + str(e[0]) + " " + str(e[1]) + ")\n")

        f.write("(property 0 int \"time\"\n")
        f.write("\t(default \"0\" \"0\")\n")
        for n in nodes_to_use:
            f.write("\t(node " + str(n) + str(" \"") + str(self.time[n]) + "\")\n")
        f.write(")\n")

        if write_layout:
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

        if node_properties:
            for p_name, (p_dict, default) in node_properties.items():
                if type(list(p_dict.values())[0]) == str:
                    f.write("(property 0 string \"%s\"\n"%p_name)
                    f.write("\t(default %s %s)\n"%(default, default))
                elif isinstance(list(p_dict.values())[0], Number):
                	f.write("(property 0 double \"%s\"\n"%p_name)
                	f.write("\t(default \"0\" \"0\")\n")
                for n in nodes_to_use:
                    f.write("\t(node " + str(n) + str(" \"") + \
                            str(p_dict.get(n, default)) + "\")\n")
                f.write(")\n")

        f.write(")")
        f.close()

    def median_average(self, subset):
        ''' Build the median vector of a subset *subset* of cells. *WARNING DEPRECATED*
            Since deprecated, no more doc.
        '''
        subset_dist = [np.mean([di.pos for di in c.D], axis = 0) - c.pos for c in subset if c.D != []]
        target_C = [c for c in subset if c.D != []]
        if subset_dist != []:
            med_distance = spatial.distance.squareform(spatial.distance.pdist(subset_dist))
            return subset_dist[np.argmin(np.sum(med_distance, axis=0))]
        else:
            return [0, 0, 0]

    def median_average_bw(self, subset):
        ''' Build the median vector of a subset *subset* of cells. *WARNING DEPRECATED*
            Since deprecated, no more doc.
        '''
        subset_dist = [c.M.pos - c.pos for c in subset if c.M != self.R]
        target_C = [c for c in subset if c.D != []]
        if subset_dist != []:
            med_distance = spatial.distance.squareform(spatial.distance.pdist(subset_dist))
            return subset_dist[np.argmin(np.sum(med_distance, axis=0))]
        else:
            return [0, 0, 0]

    def build_median_vector(self, C, dist_th, delta_t=2):
        ''' Computes the median vector for a cell *C*. *WARNING DEPRECATED*
            Since deprecated, no more doc.
        '''
        if not hasattr(self, 'spatial_edges'):
            self.compute_spatial_edges(dist_th)
        subset = [C]
        subset += C.N
        added_D = added_M = subset
        for i in range(delta_t):
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
        ''' Builds the median vectors of every nodes using the cells at a distance *dist_th*. *WARNING DEPRECATED*
            Since deprecated, no more doc.
        '''
        ruler = 0
        for C in self.nodes:
            if ruler != C.time:
                print(C.time)
            C.direction = self.build_median_vector(C, dist_th)
            ruler = C.time

    def single_cell_propagation(self, params):
        ''' Computes the incoming displacement vector of a cell. *WARNING DEPRECATED*
            Since deprecated, no more doc.
        '''
        C, t, nb_max, dist_max, to_check_self, R, pos, successor, predecessor = params
        idx3d = self.kdtrees[t]
        closest_cells = np.array(to_check_self)[list(idx3d.query(tuple(pos[C]), nb_max)[1])]
        max_value = np.min(np.where(np.array([_dist_v(pos[C], pos[ci]) for ci in closest_cells]+[dist_max+1])>dist_max))
        cells_to_keep = closest_cells[:max_value]
        subset_dist = [np.mean([pos[cii] for cii in predecessor[ci]], axis=0) - pos[ci] for ci in cells_to_keep if not ci in R]
        if subset_dist != []:
            med_distance = spatial.distance.squareform(spatial.distance.pdist(subset_dist))
            med = subset_dist[np.argmin(np.sum(med_distance, axis=0))]
        else:
            med = [0, 0, 0]
        return C, med

    def read_from_csv(self, file_path, z_mult, link=1, delim=','):
        def convert_for_csv(v):
            if v.isdigit():
                return int(v)
            else:
                return float(v)
        with open(file_path) as f:
            lines = f.readlines()
            f.close()
        self.time_nodes = {}
        self.time_edges = {}
        unique_id = 0
        self.nodes = set()
        self.edges = set()
        self.successor = {}
        self.predecessor = {}
        self.pos = {}
        self.time_id = {}
        self.time = {}
        self.lin = {}
        self.C_lin = {}
        if not link:
            self.displacement = {}
        lines_to_int = []
        corres = {}
        for l in lines:
            lines_to_int += [[convert_for_csv(v.strip()) for v in l.split(delim)]]
        lines_to_int = np.array(lines_to_int)
        if link==2:
            lines_to_int = lines_to_int[np.argsort(lines_to_int[:, 0])]
        else:
            lines_to_int = lines_to_int[np.argsort(lines_to_int[:, 1])]
        for l in lines_to_int:
            if link==1:
                id_, t, z, y, x, pred, lin_id = l
            elif link==2:
                t, z, y, x, id_, pred, lin_id = l
            else:
                id_, t, z, y, x, dz, dy, dx = l
                pred = None
            t = int(t)
            pos = np.array([x, y, z])
            C = unique_id
            corres[id_] = C
            pos[-1] = pos[-1]*z_mult
            if pred in corres:
                M = corres[pred]
                self.predecessor[C] = [M]
                self.successor.setdefault(M, []).append(C)
                self.edges.add((M, C))
                self.time_edges.setdefault(t, []).append((M, C))
                self.lin.setdefault(lin_id, []).append(C)
                self.C_lin[C] = lin_id
            self.pos[C] = pos
            self.nodes.add(C)
            self.time_nodes.setdefault(t, []).append(C)
            # self.time_id[(t, cell_id)] = C
            self.time[C] = t
            if not link:
                self.displacement[C] = np.array([dx, dy, dz*z_mult])
            unique_id += 1
        self.max_id = unique_id - 1
        self.t_b = min(self.time_nodes)
        self.t_e = max(self.time_nodes)



    def read_from_pkl_ASTEC(self, file_path, eigen=False):
        import pickle as pkl
        with open(file_path, 'rb') as f:
            tmp_data = pkl.load(f, encoding="latin1")
            f.close()
        self.name = {}
        self.volume = {}
        self.lT2pkl = {}
        self.pkl2lT = {}
        self.contact = {}
        self.prob_cells = set()
        if 'cell_lineage' in tmp_data:
            lt = tmp_data['cell_lineage']
        elif 'lin_tree' in tmp_data:
            lt = tmp_data['lin_tree']
        else:
            lt = tmp_data['Lineage tree']
        if 'cell_name' in tmp_data:
            names = tmp_data['cell_name']
        elif 'Names' in tmp_data:
            names = tmp_data['Names']
        else:
            names = {}
        if 'cell_fate' in tmp_data:
            self.fates = {}
            fates = tmp_data['cell_fate']
            do_fates = True
        else:
            do_fates = False
        if 'cell_volume' in tmp_data:
            do_volumes = True
            volumes = tmp_data['cell_volume']
        elif 'volume_information' in tmp_data:
            do_volumes = True
            volumes = tmp_data['volume_information']
        else:
            do_volumes = False
        if 'cell_contact_surface' in tmp_data:
            do_surf = True
            surfaces = tmp_data['cell_contact_surface']
        else:
            do_surf = False
        if 'problematic_cells' in tmp_data:
            prob_cells = set(tmp_data['problematic_cells'])
        else:
            prob_cells = set()
        if eigen:
            self.eigen_vectors = {}
            self.eigen_values = {}
            eig_val = tmp_data['cell_principal_values']
            eig_vec = tmp_data['cell_principal_vectors']

        inv = {vi: [c] for c, v in lt.items() for vi in v}
        nodes = set(lt).union(inv)
        if 'cell_barycenter' in tmp_data:
            pos = tmp_data['cell_barycenter']
        elif 'Barycenters' in tmp_data:
            pos = tmp_data['Barycenters']
        else:
            pos = dict(list(zip(nodes, [[0., 0., 0.], ]*len(nodes))))

        unique_id = 0
        id_corres = {}
        for n in nodes:
            if n in prob_cells:
                self.prob_cells.add(unique_id)
            # if n in pos and n in names:
            t = n // 10**4
            self.lT2pkl[unique_id] = n
            self.pkl2lT[n] = unique_id
            self.name[unique_id] = names.get(n, '')
            position = np.array(pos.get(n, [0, 0, 0]), dtype=np.float)
            self.time_nodes.setdefault(t, []).append(unique_id)
            self.nodes += [unique_id]
            self.pos[unique_id] = position
            self.time[unique_id] = t
            id_corres[n] = unique_id
            if do_volumes:
                self.volume[unique_id] = volumes.get(n, 0.)
            if eigen:
                self.eigen_vectors[unique_id] = eig_vec.get(n)
                self.eigen_values[unique_id] = eig_val.get(n)
            if do_fates:
                self.fates[unique_id] = fates.get(n, '')

            unique_id += 1
        # self.contact = {self.pkl2lT[c]: v for c, v in surfaces.iteritems() if c in self.pkl2lT}
        if do_surf:
            for c in nodes:
                if c in surfaces and c in self.pkl2lT:
                    self.contact[self.pkl2lT[c]] = {self.pkl2lT.get(n, -1): s for n, s in surfaces[c].items()
                                                                    if n%10**4==1 or n in self.pkl2lT}

        for n, new_id in id_corres.items():
            # new_id = id_corres[n]
            if n in inv:
                self.predecessor[new_id] = [id_corres[ni] for ni in inv[n]]
            if n in lt:
                self.successor[new_id] = [id_corres[ni] for ni in lt[n] if ni in id_corres]
                self.edges += [(new_id, ni) for ni in self.successor[new_id]]
                for ni in self.successor[new_id]:
                    # self.edges += [(new_id, ni)]
                    self.time_edges.setdefault(t-1, []).append((new_id, ni))

        self.t_b = min(self.time_nodes)
        self.t_e = max(self.time_nodes)
        self.max_id = unique_id

    def read_from_txt_for_celegans(self, file):
        implicit_l_t = {
            'AB': 'P0',
            'P1': 'P0',
            'EMS': 'P1',
            'P2': 'P1',
            'MS': 'EMS',
            'E': 'EMS',
            'C': 'P2',
            'P3': 'P2',
            'D': 'P3',
            'P4': 'P3',
            'Z2': 'P4',
            'Z3': 'P4'
        }

        f = open(file, 'r')
        raw = f.readlines()[1:]
        f.close()
        self.name = {}

        unique_id = 0
        for l in raw:
            t = int(l.split('\t')[0])
            self.name[unique_id] = l.split('\t')[1]
            position = np.array(l.split('\t')[2:5], dtype = np.float)
            self.time_nodes.setdefault(t, []).append(unique_id)
            self.nodes += [unique_id]
            self.pos[unique_id] = position
            self.time[unique_id] = t
            unique_id += 1

        self.t_b = min(self.time_nodes)
        self.t_e = max(self.time_nodes)

        for t, cells in self.time_nodes.items():
            if t != self.t_b:
                prev_cells = self.time_nodes[t-1]
                name_to_id = {self.name[c]: c for c in prev_cells}
                for c in cells:
                    if self.name[c] in name_to_id:
                        p = name_to_id[self.name[c]]
                    elif self.name[c][:-1] in name_to_id:
                        p = name_to_id[self.name[c][:-1]]
                    elif implicit_l_t.get(self.name[c]) in name_to_id:
                        p = name_to_id[implicit_l_t.get(self.name[c])]
                    else:
                        print('error, cell %s has no predecessors'%self.name[c])
                        p = None
                    self.predecessor.setdefault(c, []).append(p)
                    self.successor.setdefault(p, []).append(c)
                    self.edges += [(p, c)]
                    self.time_edges.setdefault(t-1, []).append((p, c))
            self.max_id = unique_id



    def read_from_xml(self, file_format, tb, te, z_mult=1., mask=None):
        ''' Reads a lineage tree from TGMM xml output.
            Args:
                file_format: string, path to the xmls location.
                        it should be written as follow:
                            path/to/xml/standard_name_t%06d.xml where (as an example)
                            %06d means a series of 6 digits representing the time and
                            if the time values is smaller that 6 digits, the missing
                            digits are filed with 0s
                tb: int, first time point to read
                te: int, last time point to read
                z_mult: float, aspect ratio
                mask: SpatialImage, binary image that specify the region to read
        '''
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
            print(t, end=' ')
            if t%10==0:
                print()
            tree = ET.parse(file_format%t)
            root = tree.getroot()
            self.time_nodes[t] = []
            self.time_edges[t] = []
            for it in root:
                if not '-1.#IND' in it.attrib['m'] and not 'nan' in it.attrib['m']:
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
                    except Exception as e:
                        pass
                else:
                    if t in self.ind_cells:
                        self.ind_cells[t] += 1
                    else:
                        self.ind_cells[t] = 1
        self.max_id = unique_id - 1

    def read_from_mamut_xml(self, path):
        ''' Read a lineage tree from a MaMuT xml.
            Args:
                path: string, path to the MaMut xml
        '''
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
                cell_id, n, x, y, z = (int(cell.attrib['ID']),
                                       cell.attrib['name'],
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
        self.track_name = {}
        for track in AllTracks:
            t_id, l = int(track.attrib['TRACK_ID']), float(track.attrib['TRACK_DURATION'])
            t_name = track.attrib['name']
            tracks[t_id] = []
            for edge in track:
                s, t = int(edge.attrib['SPOT_SOURCE_ID']), int(edge.attrib['SPOT_TARGET_ID'])
                if s in self.nodes and t in self.nodes:
                    if self.time[s] > self.time[t]:
                        s, t = t, s
                    self.successor.setdefault(s, []).append(t)
                    self.predecessor.setdefault(t, []).append(s)
                    self.track_name[s] = t_name
                    self.track_name[t] = t_name
                    tracks[t_id].append((s, t))
                    self.edges.append((s, t))
        self.t_b = min(self.time_nodes.keys())
        self.t_e = max(self.time_nodes.keys())

    def to_binary(self, fname, starting_points=None):
        ''' Writes the lineage tree (a forest) as a binary structure
            (assuming it is a binary tree, it would not work for *n*ary tree with 2 < *n*).
            The binary file is composed of 3 sequences of numbers and
            a header specifying the size of each of these sequences.
            The first sequence, *number_sequence*, represents the lineage tree
            as a DFT preporder transversal list. -1 signifying a leaf and -2 a branching
            The second sequence, *time_sequence*, represent the starting time of each tree.
            The third sequence, *pos_sequence*, reprensent the 3D coordinates of the objects.
            The header specify the size of each of these sequences.
            Each size is stored as a long long
            The *number_sequence* is stored as a list of long long (0 -> 2^(8*8)-1)
            The *time_sequence* is stored as a list of unsigned short (0 -> 2^(8*2)-1)
            The *pos_sequence* is stored as a list of double
            Args:
                fname: string, name of the binary file
                starting_points: [int, ], list of the roots to be written.
                        If None, all roots are written
                        Default: None
        '''
        if starting_points is None:
            starting_points = [c for c in self.successor.keys() if self.predecessor.get(c, []) == []]
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
                if self.successor.get(curr_c, []) == []:
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


    def read_from_binary(self, fname, reverse_time=False):
        ''' Reads a binary lineageTree file name.
            Format description:
                see self.to_binary
            Args:
                fname: string, path to the binary file
                reverse_time: bool, not used
        '''
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
        print(number_sequence[0])
        i = 0
        done = False
        if max(number_sequence[::2]) == -1:
            # number_sequence = number_sequence[1::2]
            tmp = number_sequence[1::2]
            if len(tmp)*3 == len(pos_sequence) == len(time_sequence)*3:
                time = dict(list(zip(tmp, time_sequence)))
                for c, t in time.items():
                    time_nodes.setdefault(t, []).append(c)
                pos = dict(list(zip(tmp, np.reshape(pos_sequence, (len_time, 3)))))
                is_root = {c:True for c in tmp}
                nodes = tmp
                done = True
        shown = set()
        while i < len(number_sequence) and not done:#, c in enumerate(number_sequence[:-1]):
            c = number_sequence[i]
            # if (1000.*i)//len(number_sequence)%10==0 and not (1000.*i)//len(number_sequence) in shown:
            #     shown.add((1000.*i)//len(number_sequence))
            #     sys.stdout.write('\b'*(4))
            #     sys.stdout.flush()
            #     sys.stdout.write("%03d"%((100*i)//len(number_sequence))+"%")#(100*i)//len(number_sequence)
            #     sys.stdout.flush()
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
                i += 1
                if waiting_list != []:
                    prev_mother = waiting_list.pop()
                    successor[prev_mother].insert(0, number_sequence[i+1])
                    edges.append((prev_mother, number_sequence[i+1]))
                    time_edges.setdefault(t, []).append((prev_mother, number_sequence[i+1]))
                    if i+1<len(number_sequence):
                        is_root[number_sequence[i+1]] = False
                    t = time[prev_mother] + 1
                else:
                    if 0 < len(time_sequence):
                        t = time_sequence.pop(0)
                    if i+1<len(number_sequence):
                        is_root[number_sequence[i+1]] = True
            i += 1

        predecessor = {vi: [k] for k, v in successor.items() for vi in v}
        print('100%')

        self.successor = successor
        self.predecessor = predecessor
        self.time = time
        self.time_nodes = time_nodes
        self.time_edges = time_edges
        self.pos = pos
        self.nodes = nodes
        self.edges = edges
        self.t_b = min(time_nodes.keys())
        self.t_e = max(time_nodes.keys())
        self.is_root = is_root
        self.max_id = max(self.nodes)

    def write_to_prune(self, file_format_input, file_format_output):
        ''' Useless function that reads and rewrites a TGMM files from a TGMM intput
            only keeping the objects in self.to_keep between time points 200 and 205 (mostly why this function is useless).
            Args:
                file_format_input: string, format of the input TGMM files
                file_format_output: string, format of the output TGMM files
        '''
        old_id_to_new = {}
        old_lin_to_new = {}
        lin_id = 0
        for t in range(200, 205+1):
            new_id = 0
            print(t, end=' ')
            if t%10==0:
                print()
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
                        if M_id != -1:
                            it.set('parent', M_id)
                        it.set('id', cell_id)
                        old_id_to_new[(t, cell_id)] = new_id
                        new_id += 1
            tree.write(file_format_output%t)

    def get_idx3d(self, t):
        ''' Get a 3d kdtree for the dataset at time *t*
            The  kdtree is stored in self.kdtrees[t]
            Args:
                t: int, time
            Returns:
                idx3d: kdtree, the built kdtree
                to_check_self: the correspondancy list:
                    If the query in the kdtree gives you the value i,
                    then it corresponds to the id in the tree to_check_self[i]
        '''
        to_check_self = self.time_nodes[t]
        if t not in self.kdtrees:
            data_corres = {}
            data = []
            for i, C in enumerate(to_check_self):
                data.append(tuple(self.pos[C]))
                data_corres[i] = C
            idx3d = KDTree(data)
            self.kdtrees[t] = idx3d
        else:
            idx3d = self.kdtrees[t]
        return idx3d, to_check_self

    def get_gabriel_graph(self, t):
        ''' Build the Gabriel graph of the given graph for time point *t*
            The Garbiel graph is then stored in self.Gabriel_graph.
            *WARNING: the graph is not recomputed if already computed. even if nodes were added*.
            Args:
                t: int, time
        '''
        if not hasattr(self, 'Gabriel_graph'):
            self.Gabriel_graph = {}

        if t not in self.Gabriel_graph:
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
                    delaunay_graph.setdefault(e2, set([])).add(e1)


            Gabriel_graph = {}

            for e1, neighbs in delaunay_graph.items():
                for ni in neighbs:
                    if not any([np.linalg.norm((data[ni] + data[e1])/2 - data[i])<np.linalg.norm(data[ni] - data[e1])/2
                            for i in delaunay_graph[e1].intersection(delaunay_graph[ni])]):
                        Gabriel_graph.setdefault(data_corres[e1], set()).add(data_corres[ni])
                        Gabriel_graph.setdefault(data_corres[ni], set()).add(data_corres[e1])
            # for e1, ni in delaunay_graph.iteritems():
            #     ni = list(ni)
            #     pos_e1 = data[e1]
            #     distances = np.array([self._dist_v(pos_e1, data[e2]) for e2 in ni])
            #     sorted_neighbs = list(np.array(ni)[np.argsort(distances)])
            #     tmp_pt = sorted_neighbs.pop(0)
            #     while not tmp_pt is None and len(idx3d.query_ball_point((data[tmp_pt] + pos_e1)/2., (self._dist_v(pos_e1, data[tmp_pt])/2)-10**-3))<=2:
            #         Gabriel_graph.setdefault(nodes[e1], set()).add(nodes[tmp_pt])
            #         Gabriel_graph.setdefault(nodes[tmp_pt], set()).add(nodes[e1])
            #         if sorted_neighbs != []:
            #             tmp_pt = sorted_neighbs.pop(0)
            #         else:
            #             tmp_pt = None

            self.Gabriel_graph[t] = Gabriel_graph

        return self.Gabriel_graph[t]

    def parallel_gabriel_graph_preprocess(self, nb_proc = 20):
        ''' Build the gabriel graphs for each time point. *WARNING DEPRECATED*
            Since deprecated, no more doc.
        '''
        mapping = []
        if not hasattr(self, 'Gabriel_graph'):
            self.Gabriel_graph = {}
        for t in range(self.t_b, self.t_e + 1):
            if t not in self.Gabriel_graph:
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


    def build_VF_propagation_backward(self, t_b=0, t_e=200, nb_max=20, dist_max=200, nb_proc=8):
        ''' Build the backward propagation from TGMM data. *WARNING DEPRECATED*
            Since deprecated, no more doc.
        '''
        self.VF = lineageTree(None, None, None)
        from time import time

        # Hack to allow pickling of kdtrees for multiprocessing
        # kdtree.node = kdtree.KDTree.node
        # kdtree.leafnode = kdtree.KDTree.leafnode
        # kdtree.innernode = kdtree.KDTree.innernode

        starting_cells = self.time_nodes[t_b]
        unique_id = 0
        self.VF.time_nodes = {t_b: []}
        for C in starting_cells:
            i = self.VF.get_next_id()
            self.VF.nodes.append(i)
            self.VF.time_nodes[t_b].append(i)
            self.VF.roots.append(i)
            self.VF.pos[i]=self.pos[C]

        for t in range(t_b, t_e, -1):
            tic = time()
            print(t, ': ', end=' ')
            to_check_VF = self.VF.time_nodes[t]

            idx3d, to_check_self = self.get_idx3d(t)

            self.VF.time_nodes[t-1] = []
            mapping = []

            Gabriel_graph = self.get_gabriel_graph(t)

            print('Gabriel graph built (%f s); '%(time() - tic), end=' ')

            cell_mapping_LT_VF = {}
            for C in to_check_VF:
                C_self = to_check_self[idx3d.query(self.VF.pos[C])[1]]
                cell_mapping_LT_VF[C_self] = C
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
            for C, med in out:
                C_VF = cell_mapping_LT_VF[C]
                self.VF.add_node(t-1, C_VF, self.VF.pos[C_VF] + med)

            idx3d, to_check_self = self.get_idx3d(t-1)
            to_check_VF = self.VF.time_nodes[t-1]

            print('propagation done (%f s); '%(time() - tic), end=' ')

            if to_check_self[0] not in self.spatial_density:
                self.compute_spatial_density(t-1, t-1, nb_max)

            print('Spatial density done (%f s); '%(time() - tic), end=' ')

            dist_to_VF, equivalence = idx3d.query([self.VF.pos[c] for c in to_check_VF], 1)

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
                if c1 != c2:
                    self.VF.fuse_nodes(c1, c2)

            print('Fusion done (%f s); '%(time() - tic), end=' ')

            idx3d, to_check_VF = self.VF.get_idx3d(t-1)[:2]
            dist_to_VF, equivalence = idx3d.query([self.pos[c] for c in to_check_self], 1)
            tmp = np.array([dist_to_VF[i]/self.spatial_density[c] for i, c in enumerate(to_check_self)])
            to_add = [to_check_self[i] for i in np.where(tmp>1)[0]]
            for C in to_add:
                self.VF.add_node(t-1, None, self.pos[C])

            print('Addition done (%f s); '%(time() - tic), end=' ')
            print('#cells: %d'%(len(self.VF.time_nodes[t-1])))

        self.VF.t_b = t_b
        self.VF.t_e = t_e

        return self.VF

    def get_predecessors(self, x, depth=None):
        ''' Computes the predecessors of the node *x* up to
            *depth* predecessors. The ordered list of ids is returned
            Args:
                x: int, id of the node to compute
                depth: int, maximum number of predecessors to return
            Returns:
                cycle: [int, ], list of ids, the last id is *x*
        '''
        cycle = [x]
        acc = 0
        while (len(self.successor.get(self.predecessor.get(cycle[0], [-1])[0], [])) == 1
                    and acc != depth):
            cycle.insert(0, self.predecessor[cycle[0]][0])
            acc += 1
        return cycle

    def get_successors(self, x, depth=None):
        ''' Computes the successors of the node *x* up to
            *depth* successors. The ordered list of ids is returned
            Args:
                x: int, id of the node to compute
                depth: int, maximum number of predecessors to return
            Returns:
                cycle: [int, ], list of ids, the first id is *x*
        '''
        cycle = [x]
        acc = 0
        while (len(self.successor.get(cycle[-1], [])) == 1
                    and acc != depth):
            cycle += self.successor[cycle[-1]]
            acc += 1
        return cycle

    def get_cycle(self, x, depth=None, depth_pred=None, depth_succ=None):
        ''' Computes the predecessors and successors of the node *x* up to
            *depth_pred* predecessors plus *depth_succ* successors.
            If the value *depth* is provided and not None,
            *depth_pred* and *depth_succ* are overwrited by *depth*.
            The ordered list of ids is returned
            Args:
                x: int, id of the node to compute
                depth: int, maximum number of predecessors and successor to return
                depth_pred: int, maximum number of predecessors to return
                depth_succ: int, maximum number of successors to return
            Returns:
                cycle: [int, ], list of ids
        '''
        if depth is not None:
            depth_pred = depth_succ = depth
        return (self.get_predecessors(x, depth_pred)[:-1] + self.get_successors(x, depth_succ))

    def compute_spatial_density(self, t_b=0, t_e=200, n_size=10):
        ''' Computes the average distance between the *n_size* closest object for a set of time points
            The results is stored in self.spatial_density
            Args:
                t_b: int, starting time to look at
                t_e: int, ending time to look at
                n_size: int, number of neighbors to look at
        '''
        time_range = [t for t in list(self.time_nodes.keys()) if t_b <= t <= t_e]
        for t in time_range:
            Cs = self.time_nodes[t]
            data_corres = {}
            data = []
            for i, C in enumerate(Cs):
                data.append(tuple(self.pos[C]))
                data_corres[i] = C
            if t not in self.kdtrees:
                idx3d = KDTree(data)
            else:
                idx3d = self.kdtrees[t]
            distances, indices = idx3d.query(data, n_size)
            self.spatial_density.update(dict(list(zip(Cs, np.mean(distances[:, 1:], axis=1)))))

    def compute_spatial_edges(self, th=50):
        ''' Innefitiently computes the connection between cells at a given distance
            Writes the output in self.spatial_edges
            Args:
                th: float, distance to consider neighbors
        '''
        self.spatial_edges=[]
        for t, Cs in self.time_nodes.items():
            nodes_tmp, pos_tmp = list(zip(*[(C, C.pos) for C in Cs]))
            nodes_tmp = np.array(nodes_tmp)
            distances = spatial.distance.squareform(spatial.distance.pdist(pos_tmp))
            nodes_to_match = np.where((0<distances) & (distances<th))
            to_link = list(zip(nodes_tmp[nodes_to_match[0]], nodes_tmp[nodes_to_match[1]]))
            self.spatial_edges.extend(to_link)
            for C1, C2 in to_link:
                C1.N.append(C2)

    def __init__(self, file_format=None, tb=None, te=None, z_mult=1., mask=None,
                 MaMuT=False, celegans=False, ASTEC=False, csv=False,
                 link=True, delim=',', eigen=False):
        ''' Main library to build tree graph representation of TGMM and SVF data
            It can read TGMM xml outputs, MaMuT files and binary files (see to_binary and read_from_binary)
            Args:
                file_format: string, either: - path format to the TGMM xml
                                             - path to the MaMuT file
                                             - path to the binary file
                tb: int, first time point (necessary for TGMM xmls only)
                te: int, last time point (necessary for TGMM xmls only)
                z_mult: float, z aspect ratio if necessary (usually only for TGMM xmls)
                mask: SpatialImage, binary image that specify the region to read (for TGMM xmls only)
                MaMuT: boolean, to specify that a MaMuT file is read
        '''
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
        if not (file_format is None or tb is None or te is None or MaMuT):
            self.read_from_xml(file_format, tb, te, z_mult=z_mult, mask = mask)
            self.t_b = tb
            self.t_e = te
        elif not (file_format is None) and MaMuT:
            self.read_from_mamut_xml(file_format)
        elif not (file_format is None) and celegans:
            self.read_from_txt_for_celegans(file_format)
        elif not (file_format is None) and ASTEC:
            self.read_from_pkl_ASTEC(file_format, eigen)
        elif not (file_format is None) and csv:
            self.read_from_csv(file_format, z_mult, link=link, delim=delim)
        elif not (file_format is None):
            self.read_from_binary(file_format)
