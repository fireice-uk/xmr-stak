function(listFilterLibs _strI _strO)
    set(_lst_in "")
    list(APPEND _lst_in ${_strI})
    list(FIND _lst_in "optimized" _msFmt)
    if(-1 EQUAL _msFmt)
        list(FIND _lst_in "debug" _msFmt)
        if(-1 EQUAL _msFmt)
            set(${_strO} "${_strI}" PARENT_SCOPE)
        endif()
    else()
        string(TOLOWER "${CMAKE_BUILD_TYPE}" _type)
        set(_grab_next FALSE)
        foreach(_ent ${_lst_in})
            if(_grab_next)
                set(${_strO} "${_ent}" PARENT_SCOPE)
                break()
            endif()
            if("${_ent}" STREQUAL "optimized")
                set(_ent "release")
            endif()
            if("${_type}" STREQUAL "${_ent}")
                set(_grab_next TRUE)
            endif()
        endforeach()
        unset(_ent)
        unset(_type)
        unset(_grab_next)
    endif()
endfunction()
# vim: et sw=4 sts=4 ts=4: