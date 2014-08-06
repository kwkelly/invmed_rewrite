-include $(PVFMM_DIR)/MakeVariables

PSC_INC = -I$(PETSC_DIR)/include -I$(PETSC_DIR)/$(PETSC_ARCH)/include 
PSC_LIB = -L$(PETSC_DIR)/lib -L$(PETSC_DIR)/$(PETSC_ARCH)/lib -lpetsc

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
	$(CXX_PVFMM) $(CXXFLAGS_PVFMM)                  $^   $(PSC_LIB) $(LDFLAGS_PVFMM) -o $@

./bin/test : $(OBJDIR)/test.o $(OBJDIR)/funcs.o 
	-@$(MKDIRS) $(dir $@)
	$(CXX_PVFMM) $(CXXFLAGS_PVFMM)                  $^   $(PSC_LIB) $(LDFLAGS_PVFMM) -o $@

$(OBJDIR)/test.o : test/test.cpp
	-@$(MKDIRS) $(dir $@)
	$(CXX_PVFMM) $(CXXFLAGS_PVFMM)                  $(PSC_INC) -I$(INCDIR) -c $< -o $@

clean:
	$(RM) -r $(BINDIR)/* $(OBJDIR)/*
	$(RM) *~ */*~
