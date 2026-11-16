#include <array>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

class Piece {
private:
    int id;
    int position;
    int orientation;

public:
    Piece(int id = 0, int pos = 0, int ori = 0)
        : id(id), position(pos), orientation(ori) {}

    void setPosition(int pos) { this->position = pos; }
    void setID(int id) { this->id = id; }
    void setOrientation(int ori) { this->orientation = ori; }

    int getID() const { return id; }
    int getPosition() const { return position; }
    int getOrientation() const { return orientation; }
};

class Cube {
private:
    std::vector<Piece> cornerList;
    std::vector<Piece> edgeList;

    // Cycles 4 corners: c[0] -> c[1] -> c[2] -> c[3] -> c[0]
    // Applies orientation twists (modulo 3) as each piece moves into its new slot
    void cycleCorners(const std::array<int, 4>& idx, const std::array<int, 4>& twist) {
        Piece temp = cornerList[idx[3]];
        for (int i = 3; i > 0; --i) {
            cornerList[idx[i]] = cornerList[idx[i - 1]];
            cornerList[idx[i]].setPosition(idx[i]);
            cornerList[idx[i]].setOrientation((cornerList[idx[i]].getOrientation() + twist[i]) % 3);
        }
        cornerList[idx[0]] = temp;
        cornerList[idx[0]].setPosition(idx[0]);
        cornerList[idx[0]].setOrientation((cornerList[idx[0]].getOrientation() + twist[0]) % 3);
    }

    // Cycles 4 edges: e[0] -> e[1] -> e[2] -> e[3] -> e[0]
    // Applies orientation flips (modulo 2) as each piece moves into its new slot
    void cycleEdges(const std::array<int, 4>& idx, const std::array<int, 4>& flip) {
        Piece temp = edgeList[idx[3]];
        for (int i = 3; i > 0; --i) {
            edgeList[idx[i]] = edgeList[idx[i - 1]];
            edgeList[idx[i]].setPosition(idx[i]);
            edgeList[idx[i]].setOrientation((edgeList[idx[i]].getOrientation() + flip[i]) % 2);
        }
        edgeList[idx[0]] = temp;
        edgeList[idx[0]].setPosition(idx[0]);
        edgeList[idx[0]].setOrientation((edgeList[idx[0]].getOrientation() + flip[0]) % 2);
    }

public:
    void addCorner(const Piece& c) { cornerList.push_back(c); }
    void addEdge(const Piece& e) { edgeList.push_back(e); }

    const std::vector<Piece>& getCorners() const { return cornerList; }
    const std::vector<Piece>& getEdges() const { return edgeList; }

    bool isValidSize() const {
        return cornerList.size() == 8 && edgeList.size() == 12;
    }

    bool isSolved() const {
        if (!isValidSize()) return false;
        for (int i = 0; i < 8; ++i) {
            if (cornerList[i].getID() != i || cornerList[i].getOrientation() != 0) {
                return false;
            }
        }
        for (int i = 0; i < 12; ++i) {
            if (edgeList[i].getID() != i || edgeList[i].getOrientation() != 0) {
                return false;
            }
        }
        return true;
    }

    // --- 6 Base Clockwise Face Turns (90 degrees) ---

    // Up (White): URF(0) -> UFL(3) -> ULB(2) -> UBR(1)
    void U() {
        cycleCorners({0, 3, 2, 1}, {0, 0, 0, 0});
        cycleEdges({0, 3, 2, 1}, {0, 0, 0, 0});
    }

    // Down (Yellow): DFR(4) -> DRB(5) -> DBL(6) -> DLF(7)
    void D() {
        cycleCorners({4, 5, 6, 7}, {0, 0, 0, 0});
        cycleEdges({4, 5, 6, 7}, {0, 0, 0, 0});
    }

    // Right (Red): URF(0) -> UBR(1) -> DRB(5) -> DFR(4)
    void R() {
        cycleCorners({0, 1, 5, 4}, {2, 1, 2, 1});
        cycleEdges({0, 10, 4, 8}, {0, 0, 0, 0});
    }

    // Left (Orange): ULB(2) -> UFL(3) -> DLF(7) -> DBL(6)
    void L() {
        cycleCorners({2, 3, 7, 6}, {2, 1, 2, 1});
        cycleEdges({2, 9, 6, 11}, {0, 0, 0, 0});
    }

    // Front (Green): UFL(3) -> URF(0) -> DFR(4) -> DLF(7)
    void F() {
        cycleCorners({3, 0, 4, 7}, {2, 1, 2, 1});
        cycleEdges({3, 8, 7, 9}, {1, 1, 1, 1});
    }

    // Back (Blue): UBR(1) -> ULB(2) -> DBL(6) -> DRB(5)
    void B() {
        cycleCorners({1, 2, 6, 5}, {2, 1, 2, 1});
        cycleEdges({1, 11, 5, 10}, {1, 1, 1, 1});
    }

    // --- Counter-Clockwise (-90 degrees) and Double Turns (180 degrees) ---

    void U_prime() { U(); U(); U(); }
    void U2()      { U(); U(); }

    void D_prime() { D(); D(); D(); }
    void D2()      { D(); D(); }

    void R_prime() { R(); R(); R(); }
    void R2()      { R(); R(); }

    void L_prime() { L(); L(); L(); }
    void L2()      { L(); L(); }

    void F_prime() { F(); F(); F(); }
    void F2()      { F(); F(); }

    void B_prime() { B(); B(); B(); }
    void B2()      { B(); B(); }

    // Helper to apply a single move by string name (e.g., "R", "U'", "F2")
    void applyMove(const std::string& move) {
        if (move == "U") U();
        else if (move == "U'") U_prime();
        else if (move == "U2") U2();
        else if (move == "D") D();
        else if (move == "D'") D_prime();
        else if (move == "D2") D2();
        else if (move == "R") R();
        else if (move == "R'") R_prime();
        else if (move == "R2") R2();
        else if (move == "L") L();
        else if (move == "L'") L_prime();
        else if (move == "L2") L2();
        else if (move == "F") F();
        else if (move == "F'") F_prime();
        else if (move == "F2") F2();
        else if (move == "B") B();
        else if (move == "B'") B_prime();
        else if (move == "B2") B2();
    }

    void printState() const {
        std::cout << "Corners (pos -> id:ori): ";
        for (const auto& c : cornerList) {
            std::cout << c.getPosition() << "->" << c.getID() << ":" << c.getOrientation() << "  ";
        }
        std::cout << "\nEdges   (pos -> id:ori): ";
        for (const auto& e : edgeList) {
            std::cout << e.getPosition() << "->" << e.getID() << ":" << e.getOrientation() << "  ";
        }
        std::cout << "\nSolved Status: " << (isSolved() ? "SOLVED" : "SCRAMBLED") << std::endl;
    }
};

Cube buildCube(const std::string& cornerLine, const std::string& edgeLine) {
    Cube result;
    char colon;
    int id, ori;

    std::istringstream cStream(cornerLine);
    int pos = 0;
    while (cStream >> id >> colon >> ori) {
        result.addCorner(Piece(id, pos++, ori));
    }

    std::istringstream eStream(edgeLine);
    pos = 0;
    while (eStream >> id >> colon >> ori) {
        result.addEdge(Piece(id, pos++, ori));
    }

    return result;
}

int main() {
    std::ifstream file("shared_memory.txt");
    if (!file.is_open()) {
        std::cerr << "Error: Could not open shared_memory.txt. Run main.py first.\n";
        return 1;
    }

    std::string cornerLine, edgeLine;
    if (!std::getline(file, cornerLine) || !std::getline(file, edgeLine)) {
        std::cerr << "Error: shared_memory.txt is incomplete or empty.\n";
        return 1;
    }

    Cube cube = buildCube(cornerLine, edgeLine);
    if (!cube.isValidSize()) {
        std::cerr << "Error: Parsed invalid piece counts (expected 8 corners, 12 edges).\n";
        return 1;
    }

    std::cout << "Initial Scanned State:\n";
    cube.printState();

    // Quick sanity test: applying (R U R' U') 6 times returns any cube to its exact starting state
    std::cout << "\nTesting 6x (R U R' U') cycle...\n";
    for (int i = 0; i < 6; ++i) {
        cube.applyMove("R");
        cube.applyMove("U");
        cube.applyMove("R'");
        cube.applyMove("U'");
    }
    cube.printState();

    return 0;
}
