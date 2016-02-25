/*
 * CL_LC_Graph.h
 *
 *  Created on: Nov 19, 2015
 *      Author: Rashid Kaleem (rashid.kaleem@gmail.com)
 */
#include "CL_Header.h"
#include "CL_Kernel.h"
#include <boost/iterator/counting_iterator.hpp>
#ifndef GDIST_EXP_INCLUDE_OPENCL_CL_LC_Graph_H_
#define GDIST_EXP_INCLUDE_OPENCL_CL_LC_Graph_H_

namespace Galois {
   namespace OpenCL {
      namespace Graphs {
         enum GRAPH_FIELD_FLAGS {
            NODE_DATA=0x1,EDGE_DATA=0x10,OUT_INDEX=0x100, NEIGHBORS=0x1000,ALL=0x1111, ALL_DATA=0x0011, STRUCTURE=0x1100,
         };
         /*
          std::string vendor_name;
          cl_platform_id m_platform_id;
          cl_device_id m_device_id;
          cl_context m_context;
          cl_command_queue m_command_queue;
          cl_program m_program;
          */
         struct _CL_LC_Graph_GPU {
            cl_mem outgoing_index;
            cl_mem node_data;
            cl_mem neighbors;
            cl_mem edge_data;
            cl_uint num_nodes;
            cl_uint num_edges;

         };

         static const char * cl_wrapper_str_CL_LC_Graph =
         "\
      typedef struct _GraphType { \n\
   uint _num_nodes;\n\
   uint _num_edges;\n\
    uint _node_data_size;\n\
    uint _edge_data_size;\n\
    __global uint *_node_data;\n\
    __global uint *_out_index;\n\
    __global uint *_out_neighbors;\n\
    __global uint *_out_edge_data;\n\
    }GraphType;\
      ";
         static const char * init_kernel_str_CL_LC_Graph =
         "\
      __kernel void initialize_graph_struct(__global uint * res, __global uint * g_meta, __global uint *g_node_data, __global uint * g_out_index, __global uint * g_nbr, __global uint * edge_data){ \n \
      __global GraphType * g = (__global GraphType *) res;\n\
      g->_num_nodes = g_meta[0];\n\
      g->_num_edges = g_meta[1];\n\
      g->_node_data_size = g_meta[2];\n\
      g->_edge_data_size= g_meta[3];\n\
      g->_node_data = g_node_data;\n\
      g->_out_index= g_out_index;\n\
      g->_out_neighbors = g_nbr;\n\
      g->_out_edge_data = edge_data;\n\
      }\n\
      ";

         template<typename NodeDataTy, typename EdgeDataTy>
         struct CL_LC_Graph {

            typedef NodeDataTy NodeDataType;
            typedef EdgeDataTy EdgeDataType;
            typedef boost::counting_iterator<uint64_t> NodeIterator;
            typedef boost::counting_iterator<uint64_t> EdgeIterator;
            typedef unsigned int NodeIDType;
            typedef unsigned int EdgeIDType;
         protected:
            Galois::OpenCL::CLContext * ctx = getCLContext();
            //CPU Data
            size_t _num_nodes;
            size_t _num_edges;
            unsigned int _max_degree;
            const size_t SizeEdgeData;
            const size_t SizeNodeData;
            unsigned int * outgoing_index;
            unsigned int * neighbors;
            NodeDataType * node_data;
            EdgeDataType * edge_data;
            //GPU Data
            _CL_LC_Graph_GPU gpu_wrapper;
            cl_mem gpu_struct_ptr;
            cl_mem gpu_meta;

            NodeDataType * getData() {
               return node_data;
            }

         public:
/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////
            CL_LC_Graph() :
            SizeEdgeData(sizeof(EdgeDataType) / sizeof(unsigned int)), SizeNodeData(sizeof(NodeDataType) / sizeof(unsigned int)) {
//      fprintf(stderr, "Created LC_LinearArray_Graph with %d node %d edge data.", (int) SizeNodeData, (int) SizeEdgeData);
               _max_degree = _num_nodes = _num_edges = 0;
               outgoing_index=neighbors=nullptr;
               node_data =nullptr;
               edge_data = nullptr;
               gpu_struct_ptr = gpu_meta= nullptr;

            }
            template<typename GaloisGraph>
            void load_from_galois(GaloisGraph & ggraph) {
               typedef typename GaloisGraph::GraphNode GNode;
               const size_t gg_num_nodes = ggraph.size();
               const size_t gg_num_edges = ggraph.sizeEdges();
               init(gg_num_nodes, gg_num_edges);
               int edge_counter = 0;
               int node_counter = 0;
               outgoing_index[0] = 0;
               for (auto n = ggraph.begin(); n != ggraph.end(); n++, node_counter++) {
                  int src_node = *n;
                  node_data[src_node] = ggraph.getData(*n);
                  outgoing_index[src_node] = edge_counter;
                  for (auto nbr = ggraph.edge_begin(*n); nbr != ggraph.edge_end(*n); ++nbr) {
                     GNode dst = ggraph.getEdgeDst(*nbr);
                     neighbors[edge_counter] = dst;
                     edge_data[edge_counter] = ggraph.getEdgeData(*nbr);
//               std::cout<<src_node<<" "<<dst<<" "<<out_edge_data()[edge_counter]<<"\n";
                     edge_counter++;
                  }
               }
               outgoing_index[gg_num_nodes] = edge_counter;
               fprintf(stderr, "Debug :: %d %d \n", node_counter, edge_counter);
               if (node_counter != gg_num_nodes)
               fprintf(stderr, "FAILED EDGE-COMPACTION :: %d, %zu\n", node_counter, gg_num_nodes);
               assert(edge_counter == gg_num_edges && "Failed to add all edges.");
               init_graph_struct();
               fprintf(stderr, "Loaded from GaloisGraph [V=%zu,E=%zu,ND=%lu,ED=%lu].\n", gg_num_nodes, gg_num_edges, sizeof(NodeDataType), sizeof(EdgeDataType));
            }
            template<typename GaloisGraph>
            void writeback_from_galois(GaloisGraph & ggraph) {
               typedef typename GaloisGraph::GraphNode GNode;
               const size_t gg_num_nodes = ggraph.size();
               const size_t gg_num_edges = ggraph.sizeEdges();
               int edge_counter = 0;
               int node_counter = 0;
               outgoing_index[0] = 0;
               for (auto n = ggraph.begin(); n != ggraph.end(); n++, node_counter++) {
                  int src_node = *n;
//               std::cout<<*n<<", "<<ggraph.getData(*n)<<", "<< getData()[src_node]<<"\n";
                  ggraph.getData(*n) = getData()[src_node];

               }
               fprintf(stderr, "Writeback from GaloisGraph [V=%zu,E=%zu,ND=%lu,ED=%lu].\n", gg_num_nodes, gg_num_edges, sizeof(NodeDataType), sizeof(EdgeDataType));
            }

            //TODO RK : fix - might not work with changes in interface.
            template<typename GaloisGraph>
            void load_from_galois(GaloisGraph & ggraph, int gg_num_nodes, int gg_num_edges, int num_ghosts) {
               typedef typename GaloisGraph::GraphNode GNode;
               init(gg_num_nodes + num_ghosts, gg_num_edges);
               fprintf(stderr, "Loading from GaloisGraph [%d,%d,%d].\n", (int) gg_num_nodes, (int) gg_num_edges, num_ghosts);
               int edge_counter = 0;
               int node_counter = 0;
               for (auto n = ggraph.begin(); n != ggraph.begin() + gg_num_nodes; n++, node_counter++) {
                  int src_node = *n;
                  getData(src_node) = ggraph.getData(*n);
                  outgoing_index[src_node] = edge_counter;
                  for (auto nbr = ggraph.edge_begin(*n); nbr != ggraph.edge_end(*n); ++nbr) {
                     GNode dst = ggraph.getEdgeDst(*nbr);
                     neighbors[edge_counter] = dst;
                     edge_data[edge_counter] = ggraph.getEdgeData(*nbr);
                     edge_counter++;
                  }
               }
               for (; node_counter < gg_num_nodes + num_ghosts; node_counter++) {
                  outgoing_index[node_counter] = edge_counter;
               }
               outgoing_index[gg_num_nodes] = edge_counter;
               if (node_counter != gg_num_nodes)
               fprintf(stderr, "FAILED EDGE-COMPACTION :: %d, %d, %d\n", node_counter, gg_num_nodes, num_ghosts);
               assert(edge_counter == gg_num_edges && "Failed to add all edges.");
               init_graph_struct();
               fprintf(stderr, "Loaded from GaloisGraph [V=%d,E=%d,ND=%lu,ED=%lu].\n", gg_num_nodes, gg_num_edges, sizeof(NodeDataType), sizeof(EdgeDataType));
            }
            ~CL_LC_Graph() {
               deallocate();
            }
            const cl_mem &device_ptr(){
               return gpu_struct_ptr;
            }
            void read(const char * filename) {
               readFromGR(*this, filename);
            }
            NodeDataType & getData(NodeIterator nid) {
               return node_data[*nid];
            }
            unsigned int edge_begin(NodeIterator nid) {
               return outgoing_index[*nid];
            }
            unsigned int edge_end(NodeIterator nid) {
               return outgoing_index[*nid + 1];
            }
            unsigned int num_neighbors(NodeIterator node) {
               return outgoing_index[*node + 1] - outgoing_index[*node];
            }
            unsigned int & getEdgeDst(unsigned int eid) {
               return neighbors[eid];
            }
            EdgeDataType & getEdgeData(unsigned int eid) {
               return edge_data[eid];
            }
            NodeIterator begin(){
               return NodeIterator(0);
            }
            NodeIterator end(){
               return NodeIterator(_num_nodes);
            }
            size_t size() {
               return _num_nodes;
            }
            size_t sizeEdges() {
               return _num_edges;
            }
            size_t max_degree() {
               return _max_degree;
            }
            void init(size_t n_n, size_t n_e) {
               _num_nodes = n_n;
               _num_edges = n_e;
               fprintf(stderr, "Allocating NN: :%d,  , NE %d :\n", (int) _num_nodes, (int) _num_edges);
               node_data = new NodeDataType[_num_nodes];
               edge_data = new EdgeDataType[_num_edges];
               outgoing_index= new unsigned int [_num_nodes+1];
               neighbors = new unsigned int[_num_edges];
               allocate_on_gpu();
            }
            void allocate_on_gpu() {
               fprintf(stderr, "Buffer sizes : %d , %d \n", _num_nodes, _num_edges);
               int err;
               cl_mem_flags flags = 0; //CL_MEM_READ_WRITE  ;
               cl_mem_flags flags_read = 0;//CL_MEM_READ_ONLY ;

               gpu_wrapper.outgoing_index = clCreateBuffer(ctx->get_default_device()->context(), flags_read, sizeof(cl_uint) * _num_nodes + 1, outgoing_index, &err);
               CHECK_CL_ERROR(err, "Error: clCreateBuffer of SVM - 0\n");
               gpu_wrapper.node_data = clCreateBuffer(ctx->get_default_device()->context(), flags, sizeof(cl_uint) * _num_nodes, node_data, &err);
               CHECK_CL_ERROR(err, "Error: clCreateBuffer of SVM - 1\n");
               gpu_wrapper.neighbors = clCreateBuffer(ctx->get_default_device()->context(), flags_read, sizeof(cl_uint) * _num_edges, neighbors, &err);
               CHECK_CL_ERROR(err, "Error: clCreateBuffer of SVM - 2\n");
               gpu_wrapper.edge_data = clCreateBuffer(ctx->get_default_device()->context(), flags_read, sizeof(cl_uint) * _num_edges, edge_data, &err);
               CHECK_CL_ERROR(err, "Error: clCreateBuffer of SVM - 3\n");

               gpu_struct_ptr = clCreateBuffer(ctx->get_default_device()->context(), flags, sizeof(cl_uint) * 16, outgoing_index, &err);
               CHECK_CL_ERROR(err, "Error: clCreateBuffer of SVM - 4\n");

               gpu_meta = clCreateBuffer(ctx->get_default_device()->context(), flags, sizeof(cl_uint) * 8, outgoing_index, &err);
               CHECK_CL_ERROR(err, "Error: clCreateBuffer of SVM - 5\n");
               int  cpu_meta[8];
               cpu_meta[0] = _num_nodes;
               cpu_meta[1] =_num_edges;
               cpu_meta[2] =SizeNodeData;
               cpu_meta[3] =SizeEdgeData;
               err= clEnqueueWriteBuffer(ctx->get_default_device()->command_queue(), gpu_meta, CL_TRUE,0, sizeof(int)*4, cpu_meta,NULL,0,NULL);
               CHECK_CL_ERROR(err, "Error: Writing META to GPU failed- 6\n");
               init_graph_struct();
            }
            void init_graph_struct() {
#if !PRE_INIT_STRUCT_ON_DEVICE
               this->copy_to_device();
               CL_Kernel init_kernel(getCLContext()->get_default_device());
               size_t kernel_len = strlen(cl_wrapper_str_CL_LC_Graph) + strlen(init_kernel_str_CL_LC_Graph) + 1;
               char * kernel_src = new char[kernel_len];
               sprintf(kernel_src, "%s\n%s", cl_wrapper_str_CL_LC_Graph, init_kernel_str_CL_LC_Graph);
               init_kernel.init_string(kernel_src, "initialize_graph_struct");
//               init_kernel.set_arg_list(gpu_struct_ptr, gpu_meta, gpu_wrapper.node_data, gpu_wrapper.outgoing_index, gpu_wrapper.neighbors, gpu_wrapper.edge_data);
               init_kernel.set_arg_list_raw(gpu_struct_ptr, gpu_meta, gpu_wrapper.node_data, gpu_wrapper.outgoing_index, gpu_wrapper.neighbors, gpu_wrapper.edge_data);
               init_kernel.run_task();
#endif
            }
            ////////////##############################################################///////////
            ////////////##############################################################///////////
            void deallocate(void) {
               delete [] node_data;
               delete [] edge_data;
               delete [] outgoing_index;
               delete [] neighbors;

               cl_int err = clReleaseMemObject(gpu_wrapper.outgoing_index);
               CHECK_CL_ERROR(err, "Error: clReleaseMemObject of SVM - 0\n");
               err = clReleaseMemObject(gpu_wrapper.node_data );
               CHECK_CL_ERROR(err, "Error: clReleaseMemObject of SVM - 1\n");
               err= clReleaseMemObject(gpu_wrapper.neighbors);
               CHECK_CL_ERROR(err, "Error: clReleaseMemObject of SVM - 2\n");
               err= clReleaseMemObject( gpu_wrapper.edge_data );
               CHECK_CL_ERROR(err, "Error: clReleaseMemObject of SVM - 3\n");
               err= clReleaseMemObject(gpu_struct_ptr);
               CHECK_CL_ERROR(err, "Error: clReleaseMemObject of SVM - 4\n");
               err= clReleaseMemObject(gpu_meta);
               CHECK_CL_ERROR(err, "Error: clReleaseMemObject of SVM - 5\n");
            }
            /////////////////////////////////////////////////////////////////////////////////////////////
            /////////////////////////////////////////////////////////////////////////////////////////////
            //Done
            void copy_to_host(GRAPH_FIELD_FLAGS flags = GRAPH_FIELD_FLAGS::ALL) {
               int err;
               cl_command_queue queue = ctx->get_default_device()->command_queue();
               if(flags && GRAPH_FIELD_FLAGS::OUT_INDEX){
                  err = clEnqueueReadBuffer(queue, gpu_wrapper.outgoing_index, CL_TRUE, 0, sizeof(cl_uint) * _num_nodes + 1, outgoing_index, 0, NULL, NULL);
                  CHECK_CL_ERROR(err, "Error copying 0\n");
               }
               if(flags && GRAPH_FIELD_FLAGS::NODE_DATA){
                  err = clEnqueueReadBuffer(queue, gpu_wrapper.node_data, CL_TRUE, 0, sizeof(cl_uint) * _num_nodes, node_data, 0, NULL, NULL);
                  CHECK_CL_ERROR(err, "Error copying 1\n");
               }
               if(flags && GRAPH_FIELD_FLAGS::NEIGHBORS){
                  err = clEnqueueReadBuffer(queue, gpu_wrapper.neighbors, CL_TRUE, 0, sizeof(cl_uint) * _num_edges, neighbors, 0, NULL, NULL);
                  CHECK_CL_ERROR(err, "Error copying 2\n");
               }
               if(flags && GRAPH_FIELD_FLAGS::EDGE_DATA){
                  err = clEnqueueReadBuffer(queue, gpu_wrapper.edge_data, CL_TRUE, 0, sizeof(cl_uint) * _num_edges, edge_data, 0, NULL, NULL);
                  CHECK_CL_ERROR(err, "Error copying 3\n");
               }
            }
            /////////////////////////////////////////////////////////////////////////////////////////////
            /////////////////////////////////////////////////////////////////////////////////////////////
            //Done
            void copy_to_device(GRAPH_FIELD_FLAGS flags = GRAPH_FIELD_FLAGS::ALL) {
               int err;
               cl_command_queue queue = ctx->get_default_device()->command_queue();
//               fprintf(stderr, "Buffer sizes : %d , %d \n", _num_nodes, _num_edges);
               if(flags && GRAPH_FIELD_FLAGS::OUT_INDEX) {
                  err = clEnqueueWriteBuffer(queue, gpu_wrapper.outgoing_index, CL_TRUE, 0, sizeof(cl_uint) * _num_nodes + 1, outgoing_index, 0, NULL, NULL);
                  CHECK_CL_ERROR(err, "Error copying 0\n");
               }
               if(flags && GRAPH_FIELD_FLAGS::NODE_DATA) {
                  err = clEnqueueWriteBuffer(queue, gpu_wrapper.node_data, CL_TRUE, 0, sizeof(cl_uint) * _num_nodes, node_data, 0, NULL, NULL);
                  CHECK_CL_ERROR(err, "Error copying 1\n");
               }
               if(flags && GRAPH_FIELD_FLAGS::NEIGHBORS) {
                  err = clEnqueueWriteBuffer(queue, gpu_wrapper.neighbors, CL_TRUE, 0, sizeof(cl_uint) * _num_edges, neighbors, 0, NULL, NULL);
                  CHECK_CL_ERROR(err, "Error copying 2\n");
               }
               if(flags && GRAPH_FIELD_FLAGS::EDGE_DATA) {
                  err = clEnqueueWriteBuffer(queue, gpu_wrapper.edge_data, CL_TRUE, 0, sizeof(cl_uint) * _num_edges, edge_data, 0, NULL, NULL);
                  CHECK_CL_ERROR(err, "Error copying 3\n");
               }
               ////////////Initialize kernel here.
               return;
            }
            /////////////////////////////////////////////////////////////////////////////////////////////
            /////////////////////////////////////////////////////////////////////////////////////////////
            void print_graph(void) {
               std::cout << "\n====Printing graph (" << _num_nodes << " , " << _num_edges << ")=====\n";
               for (size_t i = 0; i < _num_nodes; ++i) {
                  print_node(i);
                  std::cout << "\n";
               }
               return;
            }
            /////////////////////////////////////////////////////////////////////////////////////////////
            /////////////////////////////////////////////////////////////////////////////////////////////
            void print_node(unsigned int idx) {
               if (idx < _num_nodes) {
                  std::cout << "N-" << idx << "(" << node_data[idx] << ")" << " :: [";
                  for (size_t i = outgoing_index[idx]; i < outgoing_index[idx + 1]; ++i) {
                     std::cout << " " << neighbors[i] << "(" << edge_data[i] << "), ";
                  }
                  std::cout << "]";
               }
               return;
            }
            /////////////////////////////////////////////////////////////////////////////////////////////
            /////////////////////////////////////////////////////////////////////////////////////////////
            void print_compact(void) {
               std::cout << "\nOut-index [";
               for (size_t i = 0; i < _num_nodes + 1; ++i) {
                  std::cout << " " << outgoing_index[i] << ",";
               }
               std::cout << "]\nNeigh[";
               for (size_t i = 0; i < _num_edges; ++i) {
                  std::cout << " " << neighbors[i] << ",";
               }
               std::cout << "]\nEData [";
               for (size_t i = 0; i < _num_edges; ++i) {
                  std::cout << " " << edge_data[i] << ",";
               }
               std::cout << "]";
            }

         };
//End CL_LC_Graph
      }   //Namespace Graph
   }   //Namespace OpenCL
} // Namespace Galois

#endif /* GDIST_EXP_INCLUDE_OPENCL_CL_LC_Graph_H_ */
