-include $(PVFMM_DIR)/MakeVariables

PSC_INC = -I$(PETSC_DIR)/include -I$(PETSC_DIR)/$(PETSC_ARCH)/include 
PSC_LIB = -L$(PETSC_DIR)/lib -L$(PETSC_DIR)/$(PETSC_ARCH)/lib -lpetsc
EL_INC = -I$(WORK)/packages/elemental/include
EL_LIB = -L$(WORK)/packages/elemental/lib -lEl

RM = rm -f
MKDIRS = mkdir -p

BINDIR = ./bin
SRCDIR = ./src
OBJDIR = ./obj
INCDIR = ./include
CPP_FILES := $(wildcard src/*.cpp)
OBJ_FILES := $(addprefix obj/,$(notdir $(CPP_FILES:.cpp=.o)))

TARGET_BIN = \
	$(BINDIR)/invmed


all : $(TARGET_BIN)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp 
	-@$(MKDIRS) $(dir $@)
	$(CXX_PVFMM) $(CXXFLAGS_PVFMM)                  $(PSC_INC) -I$(INCDIR) -c $< -o $@

#$(BINDIR)/%: $(OBJDIR)/%.o $()
#	-@$(MKDIRS) $(dir $@)
#	$(CXX_PVFMM) $(CXXFLAGS_PVFMM)                  $<   $(PSC_LIB) $(LDFLAGS_PVFMM) -o $@

$(TARGET_BIN): $(OBJ_FILES)
	-@$(MKDIRS) $(dir $@)
	$(CXX_PVFMM) $(CXXFLAGS_PVFMM)   								$^   $(PSC_LIB) $(LDFLAGS_PVFMM) -o $@

./bin/test : $(OBJDIR)/test.o $(OBJDIR)/funcs.o 
	-@$(MKDIRS) $(dir $@)
	$(CXX_PVFMM) $(CXXFLAGS_PVFMM) -debug -O0                 $^   $(PSC_LIB) $(EL_LIB) $(LDFLAGS_PVFMM) -o $@

$(OBJDIR)/test.o : test/test.cpp
	-@$(MKDIRS) $(dir $@)
	$(CXX_PVFMM) $(CXXFLAGS_PVFMM) -debug -O0 -std=c++11               $(PSC_INC) $(EL_INC) -I$(INCDIR) -c $< -o $@

./bin/faims : $(OBJDIR)/faims.o $(OBJDIR)/funcs.o 
	-@$(MKDIRS) $(dir $@)
	$(CXX_PVFMM) $(CXXFLAGS_PVFMM) -debug -O0                 $^   $(PSC_LIB) $(EL_LIB) $(LDFLAGS_PVFMM) -o $@

$(OBJDIR)/faims.o : test/faims.cpp
	-@$(MKDIRS) $(dir $@)
	$(CXX_PVFMM) $(CXXFLAGS_PVFMM) -debug -O0 -std=c++11               $(PSC_INC) $(EL_INC) -I$(INCDIR) -c $< -o $@


./bin/new_test : $(OBJDIR)/new_test.o
	-@$(MKDIRS) $(dir $@)
	$(CXX_PVFMM) $(CXXFLAGS_PVFMM)                  $^   $(PSC_LIB) $(LDFLAGS_PVFMM) -o $@

$(OBJDIR)/new_test.o : test/new_test.cpp
	-@$(MKDIRS) $(dir $@)
	$(CXX_PVFMM) $(CXXFLAGS_PVFMM)                  $(PSC_INC) -I$(INCDIR) -c $< -o $@

./bin/profile_test : $(OBJDIR)/profile_test.o
	-@$(MKDIRS) $(dir $@)
	$(CXX_PVFMM) $(CXXFLAGS_PVFMM)                  $^   $(PSC_LIB) $(LDFLAGS_PVFMM) -o $@

$(OBJDIR)/profile_test.o : test/profile_test.cpp
	-@$(MKDIRS) $(dir $@)
	$(CXX_PVFMM) $(CXXFLAGS_PVFMM)                  $(PSC_INC) -I$(INCDIR) -c $< -o $@

./bin/mat_test : $(OBJDIR)/mat_test.o $(OBJDIR)/funcs.o 
	-@$(MKDIRS) $(dir $@)
	$(CXX_PVFMM) $(CXXFLAGS_PVFMM)                  $^   $(PSC_LIB) $(LDFLAGS_PVFMM) -o $@

$(OBJDIR)/mat_test.o : test/mat_test.cpp
	-@$(MKDIRS) $(dir $@)
	$(CXX_PVFMM) $(CXXFLAGS_PVFMM)                  $(PSC_INC) -I$(INCDIR) -c $< -o $@

clean:
	$(RM) -r $(BINDIR)/* $(OBJDIR)/*
	$(RM) *~ */*~
