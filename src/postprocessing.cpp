#include<pybind11/pybind11.h>
#include<iostream>
namespace py = pybind11;
class SubsNode {
private:
	int id;
	SubsNode* prev; //the pointer of previous node
	SubsNode* next; //the pointer of next node
	float subs_size;
	std::vector<SubsNode*> merge_list; //the nodes merged into current node

public:
	SubsNode(int id,float subs_size) {
		this->id = id;
		this->prev = nullptr;
		this->next = nullptr;
		this->subs_size = subs_size;

	}
	SubsNode(int id, SubsNode* next, float subs_size) {
		this->id = id;
		this->next = next;
		next->setPrev(this);
		this->subs_size = subs_size;
	};
	int getID() {
		return this->id;
	}
	void setID(int id) {
		this->id = id;
	}
	float getSize() {
		return this->subs_size;
	}
	void setSize(float subs_size) {
		this->subs_size = subs_size;
	}
	SubsNode* getPrev(){
		return this->prev;
	}
	void setPrev(SubsNode* n) {
		this->prev = n;
	}
	SubsNode* getNext() {
		return this->next;
	}
	void setNext(SubsNode* n) {
		this->next = n;
	}
	void merge(SubsNode* n) {
		this->merge_list.push_back(n);
	}
	std::vector<SubsNode*> getMergeList() {
		return this->merge_list;
	}
	int getMergeListLen() {
		return this->merge_list.size();
	}
	SubsNode* getMergedNode(int i) {
		return this->merge_list[i];
	}
};

void merge_tiny_subs(SubsNode* root, float min) {
	SubsNode* node = root;
	while (node) {
		SubsNode* next = node->getNext();
		SubsNode* prev = node->getPrev();
		if (node->getSize() < min) {

			//merge node to next
			if (next) {
				//change the id of current node to next
				node->setID(next->getID());
				if (prev) {
					//change the size to the sum of both nodes
					next->setSize(node->getSize() + next->getSize());
					//change the next node of prev to the next node
					prev->setNext(next);
				}
				else {	
					//change the size to the sum of both nodes
					node->setSize(node->getSize() + next->getSize());
					//change the next node of prev to the next node
					node->setNext(next->getNext());
				}
				next->merge(node);
			}
			else {
				//merge node to prev
				//change the id of current node to prev
				node->setID(prev->getID());
				//change the size to the sum of both nodes
				prev->setSize(node->getSize() + prev->getSize());
				//change the next node of prev to null
				prev->setNext(nullptr);
				prev->merge(node);
			}
		}
		node = node->getNext();
		if (!node) {
			break;
		}
	}
};

PYBIND11_MODULE(postprocessing, m) {
	py::class_<SubsNode>(m, "SubsNode")
		.def(py::init<int, float>())
		.def(py::init<int, SubsNode*, float>())
		.def_property("id", &SubsNode::getID, &SubsNode::setID)
		.def_property("size", &SubsNode::getSize, &SubsNode::setSize)
		.def_property_readonly("prev", &SubsNode::getPrev)
		.def_property_readonly("next", &SubsNode::getNext)
		.def_property_readonly("merge_list", &SubsNode::getMergeList)
		.def("setPrev", &SubsNode::setPrev)
		.def("setNext", &SubsNode::setNext)
		.def("get_merge_list_len", &SubsNode::getMergeListLen)
		.def("getMergedNode", &SubsNode::getMergedNode);
	m.def("merge_tiny_subs", &merge_tiny_subs, "merge tiny substructures to neighbors");
}