    #include <Python.h>
    #include <stdio.h>
    #include <stdlib.h>
    #include <string.h>

    // Min/Max macros
    #define min(X,Y) (((X) < (Y)) ? (X) : (Y))
    #define max(X,Y) (((X) > (Y)) ? (X) : (Y))

    // Vietnamese alphabet (29 letters) + digits (10) = 39 unigrams
    static const Py_UCS4 vn_unigrams[] = {
        L'a', L'ă', L'â', L'b', L'c', L'd', L'đ',
        L'e', L'ê', L'g', L'h', L'i', L'k', L'l', L'm', L'n',
        L'o', L'ô', L'ơ', L'p', L'q', L'r', L's', L't',
        L'u', L'ư', L'v', L'x', L'y',
        L'0', L'1', L'2', L'3', L'4', L'5', L'6', L'7', L'8', L'9'
    };
    static const int unigram_count = sizeof(vn_unigrams)/sizeof(vn_unigrams[0]);

    // Keep original English bigrams for now (you may replace with Vietnamese-specific)
    static const char* bigrams[50] = {
        "th","he","in","er","an","re","es","on","st","nt",
        "en","at","ed","nd","to","or","ea","ti","ar","te",
        "ng","al","it","as","is","ha","et","se","ou","of",
        "le","sa","ve","ro","ra","ri","hi","ne","me","de",
        "co","ta","ec","si","ll","so","na","li","la","el"
    };

    // Total feature size: 39*14 (levels 2–5) + 2*50 = 646
    #define PHOC_SIZE 646

    static PyObject* build_phoc(PyObject* self, PyObject* args)
    {
        PyObject* unicode_obj = NULL;
        if (!PyArg_ParseTuple(args, "U", &unicode_obj)) {
            return PyErr_Format(PyExc_RuntimeError,
                "Failed to parse argument. Call build_phoc with a Unicode string.");
        }

        // Convert to UCS4 (array of Unicode code points)
        Py_ssize_t length;
        Py_UCS4* word = PyUnicode_AsUCS4Copy(unicode_obj);
        if (!word) return NULL;
        length = PyUnicode_GetLength(unicode_obj);

        float phoc[PHOC_SIZE] = {0.0};

        // Iterate over characters
        for (int index=0; index < length; index++) {
            float char_occ0 = (float)index / (float)length;
            float char_occ1 = (float)(index+1) / (float)length;

            // Find unigram index
            int char_index = -1;
            for (int k=0; k<unigram_count; k++) {
                if (vn_unigrams[k] == word[index]) {
                    char_index = k;
                    break;
                }
            }
            if (char_index == -1) {
                // Unknown character, skip
                continue;
            }

            // Levels 2–5
            for (int level=2; level<6; level++) {
                for (int region=0; region<level; region++) {
                    float region_occ0 = (float)region / level;
                    float region_occ1 = (float)(region+1) / level;
                    float overlap0 = max(char_occ0, region_occ0);
                    float overlap1 = min(char_occ1, region_occ1);
                    float kkk = ((overlap1 - overlap0) / (char_occ1 - char_occ0));
                    if (kkk >= 0.5f) {
                        int sum=0;
                        for (int l=2; l<6; l++) if (l<level) sum+=l;
                        int feat_vec_index = sum * unigram_count + region * unigram_count + char_index;
                        phoc[feat_vec_index] = 1.0f;
                    }
                }
            }
        }

        // Add bigrams (still ASCII-based)
        int ngram_offset = unigram_count * 14;
        for (int i=0; i<(length-1); i++) {
            char utf8buf[9] = {0};   // buffer for 2-char bigram in UTF-8
            int written = wctomb(utf8buf, word[i]);
            if (written <= 0 || i+1 >= length) continue;
            written += wctomb(utf8buf+written, word[i+1]);
            if (written <= 0) continue;

            int ngram_index = -1;
            for (int k=0; k<50; k++) {
                if (strncmp(bigrams[k], utf8buf, 2) == 0) {
                    ngram_index = k;
                    break;
                }
            }
            if (ngram_index == -1) continue;

            float ngram_occ0 = (float)i / length;
            float ngram_occ1 = (float)(i+2) / length;
            int level=2;
            for (int region=0; region<level; region++) {
                float region_occ0 = (float)region / level;
                float region_occ1 = (float)(region+1) / level;
                float overlap0 = max(ngram_occ0, region_occ0);
                float overlap1 = min(ngram_occ1, region_occ1);
                if ((overlap1 - overlap0) / (ngram_occ1 - ngram_occ0) >= 0.5f) {
                    phoc[ngram_offset + region*50 + ngram_index] = 1.0f;
                }
            }
        }

        PyMem_Free(word);

        // Return as Python list
        PyObject* dlist = PyList_New(PHOC_SIZE);
        for (int i=0; i<PHOC_SIZE; i++) {
            PyList_SetItem(dlist, i, PyFloat_FromDouble((double)phoc[i]));
        }

        return dlist;
    }

    // Module defs
    static PyMethodDef myMethods[] = {
        { "build_phoc", build_phoc, METH_VARARGS, "" },
        { NULL, NULL, 0, NULL }
    };

static struct PyModuleDef cphoc_vn_module = {
    PyModuleDef_HEAD_INIT,
    "cphoc_vn",                 // must match Extension name
    "Vietnamese PHOC Module",
    -1,
    myMethods
};

PyMODINIT_FUNC PyInit_cphoc_vn(void)   // must match Extension name
{
    return PyModule_Create(&cphoc_vn_module);
}
